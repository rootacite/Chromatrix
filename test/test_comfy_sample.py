"""The ComfyUI sampling path: the bundled graph, the staged LoRA and one render pass.

Importing `trainer.generate_sample` pulls torch in (the pass that runs a spec is the same process
either way); nothing here touches the GPU — the renderer talks to a stub HTTP server.
"""

import json
import sys
import tempfile
import threading
import unittest
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest import mock

from PIL import Image

# `python test/test_comfy_sample.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import comfy, comfy_sample, genjob
from trainer import generate_sample as generator
from trainer.config import SampleSet


def _png_bytes(size=(8, 8)) -> bytes:
    """A real PNG: the runner reads ComfyUI's answer back with PIL to attach its provenance."""
    buffer = BytesIO()
    Image.new("RGB", size, (200, 100, 50)).save(buffer, "PNG")
    return buffer.getvalue()


def sample_set(**overrides) -> SampleSet:
    values = dict(
        name="a",
        prompt="pa",
        negative="na",
        width=1152,
        height=768,
        steps=35,
        guidance_scale=6.0,
        guidance_rescale=0.6,
        seed=4242,
        repeat=2,
    )
    values.update(overrides)
    return SampleSet(**values)


class StubSamplingComfy:
    """A ComfyUI that answers the calls one sampling pass makes.

    `/prompt` records the workflow it was handed (and can hang instead of finishing, so a cancel
    can be exercised), `/history/<id>` reports the SaveImage output, `/view` hands back a PNG.
    """

    def __init__(self, hang: bool = False, fail: bool = False) -> None:
        self.hang = hang
        self.fail = fail
        self.queued: list[dict] = []
        self.history: dict[str, dict] = {}
        self.progress_payload: dict = {}
        self.interrupts = 0

        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep the test output clean
                pass

            def _send(self, payload: bytes, status: int = 200, content_type: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):  # noqa: N802 - http.server's naming
                path = self.path.split("?")[0]
                if path == "/progress":
                    self._send(json.dumps(stub.progress_payload).encode())
                    return
                if path.startswith("/history/"):
                    prompt_id = path[len("/history/"):]
                    entry = stub.history.get(prompt_id)
                    self._send(json.dumps({prompt_id: entry} if entry else {}).encode())
                    return
                if path == "/view":
                    self._send(_png_bytes(), content_type="image/png")
                    return
                self._send(b"{}", status=404)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.path == "/interrupt":
                    stub.interrupts += 1
                    self._send(b"{}")
                    return
                if self.path != "/prompt":
                    self._send(b"{}", status=404)
                    return
                workflow = (json.loads(body).get("prompt")) or {}
                stub.queued.append(workflow)
                prompt_id = f"stub-{uuid.uuid4().hex[:8]}"
                if not stub.hang:
                    stub.history[prompt_id] = {
                        "status": {
                            "completed": True,
                            "status_str": "error" if stub.fail else "success",
                            "messages": [["execution_error", {}]] if stub.fail else [],
                        },
                        "outputs": (
                            {}
                            if stub.fail
                            else {"222": {"images": [{"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}]}}
                        ),
                    }
                self._send(json.dumps({"prompt_id": prompt_id}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class SamplingWorkflowTest(unittest.TestCase):
    """The bundled graph and the overrides written into it."""

    def test_the_bundled_workflow_has_every_node_an_override_writes(self):
        workflow = comfy_sample.automation.load_workflow(comfy_sample.sampling_workflow_path())
        comfy_sample.require_sampling_nodes(workflow)
        # The path is the repo's own bundled file, not a user setting.
        self.assertEqual("Sampling.json", comfy_sample.sampling_workflow_path().name)
        self.assertEqual("beta", comfy_sample.sampling_workflow_path().parent.name)

    def test_a_workflow_without_the_lora_node_is_refused(self):
        workflow = comfy_sample.automation.load_workflow(comfy_sample.sampling_workflow_path())
        del workflow[comfy_sample.SAMPLING_LORA_NODE]
        with self.assertRaises(comfy.ComfyError) as ctx:
            comfy_sample.require_sampling_nodes(workflow)
        self.assertIn(comfy_sample.SAMPLING_LORA_NODE, str(ctx.exception))

    def test_an_input_an_override_needs_is_required(self):
        workflow = comfy_sample.automation.load_workflow(comfy_sample.sampling_workflow_path())
        del workflow[comfy_sample.SAMPLING_LATENT_NODE]["inputs"]["width"]
        with self.assertRaises(comfy.ComfyError) as ctx:
            comfy_sample.require_sampling_nodes(workflow)
        self.assertIn("width", str(ctx.exception))

    def test_every_value_of_a_sample_set_reaches_its_node(self):
        workflow = comfy_sample.automation.load_workflow(comfy_sample.sampling_workflow_path())
        comfy_sample.apply_sampling(
            workflow,
            prompt="a girl",
            negative="bad",
            steps=21,
            cfg=5.5,
            rescale=0.7,
            width=640,
            height=960,
            seed=99,
            lora_name="axl-sample/job.safetensors",
            strength=0.8,
            checkpoint_name="base.safetensors",
        )
        self.assertEqual("a girl", workflow[comfy_sample.SAMPLING_POSITIVE_NODE]["inputs"]["text"])
        self.assertEqual("bad", workflow[comfy_sample.SAMPLING_NEGATIVE_NODE]["inputs"]["text"])
        sampler = workflow[comfy_sample.SAMPLING_SAMPLER_NODE]["inputs"]
        self.assertEqual((21, 5.5), (sampler["steps"], sampler["cfg"]))
        self.assertEqual(99, workflow[comfy_sample.SAMPLING_SEED_NODE]["inputs"]["seed"])
        latent = workflow[comfy_sample.SAMPLING_LATENT_NODE]["inputs"]
        self.assertEqual((640, 960), (latent["width"], latent["height"]))
        # The set's own `guidance_rescale` is RescaleCFG's multiplier, 0 = no rescale.
        self.assertEqual(0.7, workflow[comfy_sample.SAMPLING_RESCALE_NODE]["inputs"]["multiplier"])
        lora = workflow[comfy_sample.SAMPLING_LORA_NODE]["inputs"]
        self.assertEqual("axl-sample/job.safetensors", lora["lora_name"])
        self.assertEqual((0.8, 0.8), (lora["strength_model"], lora["strength_clip"]))
        self.assertEqual(
            "base.safetensors",
            workflow[comfy_sample.SAMPLING_CHECKPOINT_NODE]["inputs"]["ckpt_name"],
        )

    def test_a_blank_base_model_keeps_the_workflows_own(self):
        workflow = comfy_sample.automation.load_workflow(comfy_sample.sampling_workflow_path())
        original = workflow[comfy_sample.SAMPLING_CHECKPOINT_NODE]["inputs"]["ckpt_name"]
        comfy_sample.apply_sampling(
            workflow,
            prompt="p",
            negative="",
            steps=1,
            cfg=1.0,
            rescale=0.0,
            width=64,
            height=64,
            seed=1,
            lora_name="l.safetensors",
            strength=1.0,
        )
        self.assertEqual(original, workflow[comfy_sample.SAMPLING_CHECKPOINT_NODE]["inputs"]["ckpt_name"])

    def test_the_sampler_reads_the_bundled_workflow(self):
        sampler = comfy_sample.open_sampler(
            "http://127.0.0.1:8188", lora_name="l.safetensors", strength=1.0
        )
        self.assertEqual({"222"}, sampler._save_nodes)

    def test_a_strength_outside_the_range_is_refused(self):
        self.assertEqual(0.95, comfy_sample.normalize_strength(None))
        self.assertEqual(0.95, comfy_sample.normalize_strength(""))
        self.assertEqual(1.5, comfy_sample.normalize_strength("1.5"))
        for bad in ("nope", -0.1, 2.5, True):
            with self.assertRaises(ValueError):
                comfy_sample.normalize_strength(bad)

    def test_a_blank_base_model_reads_as_empty(self):
        self.assertEqual("", comfy_sample.normalize_checkpoint_name(None))
        self.assertEqual("wai.safetensors", comfy_sample.normalize_checkpoint_name(" wai.safetensors "))


class StageLoraTest(unittest.TestCase):
    """The copy ComfyUI loads by name, and the removal that follows every pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.install = Path(self.tmp.name) / "comfy"
        (self.install / "models" / "loras").mkdir(parents=True)
        self.checkpoint = Path(self.tmp.name) / "rein.safetensors"
        self.checkpoint.write_bytes(b"lora-bytes")
        self.env = mock.patch.object(
            comfy_sample.comfy,
            "loras_for_server",
            return_value={"root": str(self.install), "loras": [], "error": ""},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_the_copy_lands_under_the_stage_folder_and_is_named_by_the_job(self):
        staged = comfy_sample.stage_lora("http://127.0.0.1:8188", self.checkpoint, "rein_sets_gen_1")
        self.assertEqual("axl-sample/rein_sets_gen_1.safetensors", staged.name)
        self.assertEqual(b"lora-bytes", staged.path.read_bytes())
        self.assertEqual(
            self.install / "models" / "loras" / "axl-sample" / "rein_sets_gen_1.safetensors",
            staged.path,
        )

        comfy_sample.unstage_lora(staged)
        self.assertFalse(staged.path.exists())

    def test_two_passes_of_one_checkpoint_never_share_a_name(self):
        first = comfy_sample.stage_lora("http://127.0.0.1:8188", self.checkpoint, "job_a")
        second = comfy_sample.stage_lora("http://127.0.0.1:8188", self.checkpoint, "job_b")
        self.addCleanup(comfy_sample.unstage_lora, first)
        self.addCleanup(comfy_sample.unstage_lora, second)
        self.assertNotEqual(first.name, second.name)
        self.assertTrue(first.path.exists() and second.path.exists())

    def test_a_service_that_lists_no_lora_folder_is_refused(self):
        with mock.patch.object(
            comfy_sample.comfy, "loras_for_server",
            return_value={"root": "", "loras": [], "error": "no process is listening on port 8188"},
        ):
            with self.assertRaises(comfy.ComfyError) as ctx:
                comfy_sample.stage_lora("http://127.0.0.1:8188", self.checkpoint, "job")
        self.assertIn("no process is listening", str(ctx.exception))

    def test_unstaging_a_missing_file_is_not_a_failure(self):
        staged = comfy_sample.StagedLora(name="axl-sample/gone.safetensors", path=Path(self.tmp.name) / "gone")
        comfy_sample.unstage_lora(staged)
        comfy_sample.unstage_lora(None)


class ComfySamplerTest(unittest.TestCase):
    """One queued prompt: what it sends, what it reports and how it stops."""

    def setUp(self):
        self.stub = StubSamplingComfy()
        self.addCleanup(self.stub.close)

    def sampler(self, **kwargs) -> comfy_sample.ComfySampler:
        return comfy_sample.open_sampler(
            self.stub.url, lora_name="axl-sample/job.safetensors", strength=0.75, **kwargs
        )

    def test_render_queues_the_set_and_returns_the_image(self):
        sampler = self.sampler()
        data = sampler.render(sample_set(), 4242)
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertEqual(1, len(self.stub.queued))
        workflow = self.stub.queued[0]
        self.assertEqual("pa", workflow[comfy_sample.SAMPLING_POSITIVE_NODE]["inputs"]["text"])
        self.assertEqual("na", workflow[comfy_sample.SAMPLING_NEGATIVE_NODE]["inputs"]["text"])
        self.assertEqual(35, workflow[comfy_sample.SAMPLING_SAMPLER_NODE]["inputs"]["steps"])
        self.assertEqual(6.0, workflow[comfy_sample.SAMPLING_SAMPLER_NODE]["inputs"]["cfg"])
        self.assertEqual(4242, workflow[comfy_sample.SAMPLING_SEED_NODE]["inputs"]["seed"])
        lora = workflow[comfy_sample.SAMPLING_LORA_NODE]["inputs"]
        self.assertEqual("axl-sample/job.safetensors", lora["lora_name"])
        self.assertEqual(0.75, lora["strength_model"])

    def test_a_failed_prompt_raises_with_the_comfy_message(self):
        self.stub.fail = True
        with self.assertRaises(comfy.ComfyError):
            self.sampler().render(sample_set(), 1)

    def test_the_step_progress_comes_from_the_progress_endpoint(self):
        reported: list[tuple[int, int]] = []
        self.stub.hang = True  # `/history` stays empty, so the wait has to poll and report
        sampler = self.sampler()

        def on_progress(value: int, maximum: int) -> None:
            reported.append((value, maximum))

        # `/history` never completes here, so the wait is stopped by the callback itself.
        def stop() -> bool:
            return bool(reported)

        original = sampler.client.history

        def history(prompt_id: str) -> dict:
            self.stub.progress_payload = {"value": 7, "max": 35, "prompt_id": prompt_id, "node": "19"}
            return original(prompt_id)

        sampler.poll_interval = 0.01

        sampler.client.history = history  # type: ignore[method-assign]
        with self.assertRaises(comfy.ComfyCancelled):
            sampler.render(sample_set(), 1, should_stop=stop, on_progress=on_progress)
        self.assertEqual([(7, 35)], reported)

    def test_progress_from_another_prompt_is_ignored(self):
        sampler = self.sampler()
        self.stub.progress_payload = {"value": 3, "max": 35, "prompt_id": "someone-else", "node": "19"}
        self.assertIsNone(sampler._current_step("stub-mine"))

    def test_a_stop_request_ends_the_wait(self):
        self.stub.hang = True
        with self.assertRaises(comfy.ComfyCancelled):
            self.sampler().render(sample_set(), 1, should_stop=lambda: True)

    def test_interrupt_asks_comfy_to_stop_its_execution(self):
        sampler = self.sampler()
        sampler.interrupt()
        self.assertEqual(1, self.stub.interrupts)


class ComfyPassRunnerTest(unittest.TestCase):
    """`generate_sample` on a `comfy` spec: the same files, names and counters as a local pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.generated = genjob.generated_dir(Path(self.tmp.name) / "rein_samples")
        self.generated.mkdir(parents=True)
        self.install = Path(self.tmp.name) / "comfy"
        (self.install / "models" / "loras").mkdir(parents=True)
        self.checkpoint = Path(self.tmp.name) / "rein_s000100" / "rein.safetensors"
        self.checkpoint.parent.mkdir(parents=True)
        from safetensors.torch import save_file

        save_file({}, str(self.checkpoint), metadata={"ss_steps": "100", "ss_network_dim": "16"})
        self.stub = StubSamplingComfy()
        self.addCleanup(self.stub.close)
        self.env = mock.patch.object(
            comfy_sample.comfy,
            "loras_for_server",
            return_value={"root": str(self.install), "loras": [], "error": ""},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def spec(self, **overrides) -> dict:
        sets = [
            sample_set(name="a", prompt="pa", seed=100, repeat=2),
            sample_set(name="b", prompt="pb", seed=200, repeat=1, width=768, height=1152),
        ]
        job = genjob.new_job(
            {"step": 100},
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint=str(self.checkpoint),
            mode=genjob.MODE_SETS,
            total_images=3,
            backend=genjob.BACKEND_COMFY,
            comfy={
                "server": self.stub.url,
                "checkpoint": "waiIllustrated.safetensors",
                "strength": 0.8,
                "workflow": str(comfy_sample.sampling_workflow_path()),
            },
            extra={"sample_sets": [asdict(entry) for entry in sets]},
        )
        job.update(overrides)
        genjob.write_job(self.generated, job)
        return job

    def test_the_pass_writes_one_image_per_slot_and_removes_the_staged_lora(self):
        job = self.spec()
        generator.run_sample_sets(job, self.generated)

        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        self.assertEqual(genjob.STATE_DONE, stored["state"])
        self.assertEqual(3, stored["images_done"])
        self.assertEqual(
            sorted(f"{job['id']}_p{index}_{repeat}.png" for index, repeat in ((0, 0), (0, 1), (1, 0))),
            sorted(Path(path).name for path in stored["files"]),
        )
        for path in stored["files"]:
            self.assertTrue(Path(path).is_file())
        # Each slot's own set: the prompts and values the run samples with, not the graph's.
        self.assertEqual(
            ["pa", "pa", "pb"],
            [w[comfy_sample.SAMPLING_POSITIVE_NODE]["inputs"]["text"] for w in self.stub.queued],
        )
        self.assertEqual(
            [100, 101, 200],
            [w[comfy_sample.SAMPLING_SEED_NODE]["inputs"]["seed"] for w in self.stub.queued],
        )
        self.assertEqual(
            [(1152, 768), (1152, 768), (768, 1152)],
            [
                (
                    w[comfy_sample.SAMPLING_LATENT_NODE]["inputs"]["width"],
                    w[comfy_sample.SAMPLING_LATENT_NODE]["inputs"]["height"],
                )
                for w in self.stub.queued
            ],
        )
        # A pass stages exactly one copy and takes it away again.
        self.assertEqual([], list((self.install / "models" / "loras" / "axl-sample").glob("*")))

    def test_the_written_image_carries_the_comfy_provenance(self):
        job = self.spec()
        generator.run_sample_sets(job, self.generated)
        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        from trainer import provenance

        record = provenance.read_provenance(stored["files"][0])
        self.assertEqual("100", record[f"{provenance.PREFIX}seed"])
        self.assertEqual("pa", record[f"{provenance.PREFIX}prompt"])
        self.assertEqual("sets", record[f"{provenance.PREFIX}source"])
        self.assertEqual("comfy", record[f"{provenance.PREFIX}backend"])
        self.assertEqual("waiIllustrated.safetensors", record[f"{provenance.PREFIX}comfy_checkpoint"])
        self.assertEqual("0.8", record[f"{provenance.PREFIX}comfy_lora_strength"])
        self.assertEqual("16", record[f"{provenance.PREFIX}network_dim"])

    def test_a_plan_renders_only_its_slots(self):
        job = self.spec()
        rendered = generator._render_sets_comfy(
            spec=job,
            sets=generator._record_sets(job),
            generated=self.generated,
            job_id=str(job["id"]),
            checkpoint=self.checkpoint,
            plan=[(1, 0)],
        )
        self.assertEqual(1, len(rendered))
        self.assertEqual((1, 0), rendered[0][:2])
        self.assertEqual(1, len(self.stub.queued))

    def test_a_job_without_a_server_is_refused_before_anything_is_staged(self):
        job = self.spec()
        job["comfy"] = {**job["comfy"], "server": ""}
        with self.assertRaises(RuntimeError) as ctx:
            generator.run_sample_sets(job, self.generated)
        self.assertIn("no ComfyUI server", str(ctx.exception))

    def test_a_comfy_failure_lands_on_the_job_as_an_error(self):
        job = self.spec()
        self.stub.fail = True
        with self.assertRaises(RuntimeError):
            generator.run_sample_sets(job, self.generated)
        self.assertEqual([], list((self.install / "models" / "loras" / "axl-sample").glob("*")))

    def test_the_batch_child_records_what_rendered_it(self):
        batch = genjob.new_batch_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoints=[{"path": str(self.checkpoint), "step": 100, "dir": str(self.checkpoint.parent)}],
            from_step=100,
            to_step=100,
            images_per_checkpoint=3,
            sample_sets=[asdict(sample_set(name="a", prompt="pa", seed=100, repeat=2))],
            backend=genjob.BACKEND_COMFY,
            comfy={
                "server": self.stub.url,
                "checkpoint": "",
                "strength": 0.8,
                "workflow": str(comfy_sample.sampling_workflow_path()),
            },
        )
        genjob.write_job(self.generated, batch)
        generator.run_sample_batch(batch, self.generated)

        stored = genjob.read_job(genjob.job_path(self.generated, str(batch["id"])))
        self.assertEqual(genjob.STATE_DONE, stored["state"])
        self.assertEqual(genjob.BACKEND_COMFY, stored["backend"])
        self.assertEqual(1, len(stored["job_ids"]))
        child = genjob.read_job(genjob.job_path(self.generated, stored["job_ids"][0]))
        self.assertEqual(genjob.BACKEND_COMFY, child["backend"])
        self.assertEqual(self.stub.url, child["comfy"]["server"])
        self.assertEqual(genjob.STATE_DONE, child["state"])
        self.assertEqual(2, child["images_done"])
        self.assertEqual([], list((self.install / "models" / "loras" / "axl-sample").glob("*")))


class ComfyBackendRecordTest(unittest.TestCase):
    """The record vocabulary the runner reads back."""

    def test_a_missing_backend_reads_as_the_built_in_path(self):
        self.assertFalse(generator._renders_on_comfy({}))
        self.assertFalse(generator._renders_on_comfy({"backend": genjob.BACKEND_BUILTIN}))
        self.assertTrue(generator._renders_on_comfy({"backend": genjob.BACKEND_COMFY}))

    def test_an_unknown_backend_is_refused_at_plan_time(self):
        with self.assertRaises(ValueError):
            genjob.normalize_backend("flux")

    def test_a_comfy_block_is_normalized_and_only_present_for_comfy(self):
        self.assertEqual({"backend": "builtin"}, genjob.comfy_fields("", None))
        fields = genjob.comfy_fields(
            "comfy", {"server": "http://127.0.0.1:8188", "checkpoint": "wai.safetensors", "strength": 0.5}
        )
        self.assertEqual("http://127.0.0.1:8188", fields["comfy"]["server"])
        self.assertEqual(0.5, fields["comfy"]["strength"])
        self.assertEqual("", fields["comfy"]["workflow"])


if __name__ == "__main__":
    unittest.main()
