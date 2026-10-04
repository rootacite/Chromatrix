# Dataset contract — Chromatrix

> Detail behind `AGENT.md` §8. `AGENT.md` keeps the condensed rules; this file carries the full text.


Sidecar captions, comma-separated tags, extensions: jpg/jpeg/png/webp/bmp. The dataset is the list of `[[environment.train_data]]` folders (`train_data_dir` alone when there is no list), each with its own per-epoch `repeat`; records stay unique per image and only the bucket lists carry the repeats (see `config-contract.md`). Optional loss mask: `{stem}.mask.png` next to `{stem}.png` (always PNG). If the sidecar exists, MSE is weighted by that mask (white=train, black=ignore). If it is missing and the training image has an alpha channel, that alpha is the mask (0=ignore, 255=train). Either way the letterbox pad is weight 0, so **every** sample returns a full-bucket mask from `load_loss_mask`; only sidecar/alpha images count into `n_masked` (`Loss masks: n/m samples`). A silhouette's alpha is a hard 0/255 step, so `tools/mask_blur.py` batch-writes that sidecar from the alpha instead: a Gaussian blur of it, `--radius` in source-image pixels, same size as the training image (the loader resizes a sidecar to the source size NEAREST-first, so any other size would come back stepped), marked with an `axl_mask_blur` PNG text chunk. A sidecar without that chunk (a mask painted in Chromatrix) is left alone unless `--overwrite`; the training images are never written. **Exclude** `*.mask.png` from every image listing (`list_images`, Chromatrix Images/Statistics, `agent.py`, `tagger/`). Drop/trash moves the sidecar with the pair.

Python trainer `list_images` / Chromatrix / `agent.py` should stay consistent on extensions and “same stem” pairing. Chromatrix `parse_tags` = split `,` → trim → drop empty. Duplicates preserved in captions; stats dedupe per file.

Latent cache: one per dataset folder, `<folder>/.latents_cache/{sha1(abs_path::bucket_w x bucket_h::left,top,fit_w x fit_h)}.pt`. The key carries the absolute image path, so folders never collide, and the fit geometry, so any change to the bucket rule, the clamps or the fit path forces a one-time re-encode instead of silently serving latents built from differently placed pixels (a mask edit does *not* move the key — masks are not cached). A file is a hit only when it holds the keyed bucket's latent (4 channels, spatial / 8): an unreadable file, or one holding anything else — a stray write, a half-written file left by a killed run — is a miss, warned on stderr and re-encoded over, and `warm_latent_cache` skips only what the dataset itself calls a hit (its verify pass applies the same verdict to the files already on disk, and re-encodes the ones that fail it). What a file holds is a function of (key, seed): the posterior sample is drawn per image from a generator seeded by `sha1(f"{cfg.seed}:{key}")`, not from the global one, and the encode batches are planned before any pixel is read, so neither the batch an image lands in nor the order the CPU threads finish in can reach its value — and a pass that has to encode no longer shifts the generator training draws its noise, timesteps and dropout masks from. Changing this scheme changes every value it writes without changing any key, so an existing cache keeps serving its old entries (a run reads either the old or the new bytes, never a mix within one key); bump the key if a cache has to be rebuilt. Do not hand-edit cache files.

`tagger2/` is the captioner Chromatrix runs (`python tagger2/main.py DIR --threshold 0.35`): the published Pixai tagger v1 (1008×1008 ViTDet, 30 877 Danbooru tags in six categories, its own `TaggerPipeline` via `trust_remote_code`). The weights live in the HF cache; nothing in it reaches the Hub — the script resolves the snapshot cache-only and hands the pipeline that directory, and `--download` is the one-time fetch (10.9 s / 3 images / **zero** outbound connections measured, against 16.80 s and one connection for the repository-id form). `--categories` (default `general`) picks what a caption takes; the slider is a floor over the model's own per-category `category_best_threshold`, applied by the pipeline's `min_threshold`. Underscores become spaces so captions match the legacy tagger's. `--only-tags T1,T2` is the **partial** pass: add-only, it appends those tags to the captions that show them and leaves every other tag alone (a caption that would gain nothing is not rewritten), `--categories` does not take part, and the threshold goes to the pipeline as `threshold` (an override) instead of `min_threshold`. The result carries `mode`/`only_tags`/`added`/`unmatched`. Chromatrix's Utils → Environment Auto-tag card drives both (a partial switch and a tag field). `miopen_cache/` is generated (both MIOpen variables are pinned there before torch loads) — do not treat it as source, and do not commit it. `tagger_info` exposes the same categories to Chromatrix.

`tag_directory` and `tag_paths` share one implementation (`_resolve_tagger` + `_tag_all`): the first lists a folder, writes a sidecar `.txt` per image and returns the summary `dataset_tag` answers with; the second tags exactly the paths it is given, writes nothing, and answers one `{path, name, tags, error}` entry per image — that is what an evaluation's tagging pass uses (it tags the run's own sample PNGs beside its own, so it must not leave sidecars behind). Both keep the batched pass with its one-image-at-a-time fallback, and `tag_paths` takes an `on_entry` callback: the evaluation persists the tags it has so far from there and raises `_Cancelled` when a cancel has arrived, so the stop lands between images. An empty path list answers `[]` without loading the model, and `caption_for` is `tag_list` joined, so a caption and the labels an evaluation compares are the same thing.

`tagger/` is the **legacy** ONNX WD-tagger (`python tagger/main.py DIR --threshold 0.35`), no longer reachable from Chromatrix. Model files sit next to the script (`model.onnx`, `selected_tags.csv`). `migraphx_cache/` is generated — do not treat as source. Keep `selected_tags.csv`: `tag_lexicon` and the Statistics tag card read it.

`tools/` scripts are mostly **in-place / destructive** (`tools/vpred_reference.py` is the exception: a read-only renderer that implements ComfyUI's own sampling recipe, for checking a sample the trainer produced against ComfyUI's code path). Prefer `ranko/tools/agent.py --dry-run` for agent-driven edits. `ui.py` is a **deprecated** Streamlit viewer; do not extend it.

## Where the validation images come from

Two sources, one contract, chosen by `custom_validation_path(cfg)`:

- **A percentage split** (the default): `select_validation` leaves `ceil(val_split_percent/100 × images)`
  whole images out of every training folder, and the passes draw from those. The held-out images keep
  their records and their place in `cache_entries()` (so the warm cache still encodes them) but their
  draws leave `self.buckets`, so the epoch shrinks by exactly that much.
- **A custom directory** (`[training].val_data_dir` non-empty): nothing is held out, and the
  directory supplies every validation image. It is read recursively with the same extensions, the
  same `{stem}.txt` captions and the same `.mask.png` / alpha rules, one record per image with
  `repeat = 1`, into `self.val_buckets` and its own `<dir>/.latents_cache` (a folder's cache is keyed
  by absolute path, so the two never collide). Training images therefore keep every draw, and the
  validation records are unreachable from the training buckets by construction — the probe in
  `test/test_validation_split.py` walks three epochs of real batches in both directions.
  A path that is not a directory, or one holding no usable image, fails the run with the path in the
  message; `dataset_counts` reports the same two reasons as `val_data_error` for the Utils form.

The two loss curves read only `val_buckets`, so switching sources changes which images are scored and
nothing else: `sample_validation_indices` (pass 1) and `fixed_validation_indices` (pass 2) are
subsets of those records either way, and `val_image_count` is the pool both passes see.

## Picking the fixed validation sample

`trainer/validation_split.py::select_diverse_subset` chooses the sample the `Val/Fixed_Loss` curve
scores, once per dataset, from the validation pool only (the held-out images, or the custom
directory's). A plain random draw routinely pairs
near-copies — a burst from one pose lands in one folder — and a curve built on near-identical images
says less than one built on images that differ.

- Signatures are 32×32 aspect-squashed RGB thumbnails (`_signature`, decoded in parallel with
  `Image.draft` for JPEGs). Aspect is squashed rather than letterboxed so every signature has one
  length; the aspect a training step sees is the bucket's business, not this picker's.
- `signature_similarity` is `tools/cmp_img.py:similarity`'s definition exactly — mean absolute pixel
  difference over RGB, 100 = identical — with the one difference a dataset folder demands:
  `cmp_img.compare_one` returns `None` for two images of different shapes, so it cannot rank a
  mixed-resolution folder at all.
- The traversal seeds at the image closest to the signature mean (a *typical* image, so an outlier
  cannot anchor the set), then repeatedly adds the candidate whose similarity to the most similar
  already-chosen image is lowest. Updating that worst case with a **maximum** (not a minimum) is what
  keeps a duplicate of a chosen image at ~100 and therefore out of the set.
- Deterministic in the folder contents: strict `>` comparisons, index tie-breaks, no RNG — a resume,
  a later run and the probe's own recomputation all agree.
- An unreadable image gets a blank signature and stays out of the candidates (a blank vector is
  "maximally dissimilar" and would otherwise win every comparison); it is counted in the stats the
  run logs. `count >= len(paths)` degenerates to all of them.

