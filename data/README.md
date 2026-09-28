# Workload vectors

`workload_e5_1024.npz` (numpy `savez`, float32):

| array | shape | content |
|---|---|---|
| `base` | 1069 x 1024 | e5-large passage embeddings of the evidence units of a **fully synthetic** company corpus (generated text: 170 raw chunks from docs / chats / meetings / tickets, 184 wiki chunks, 715 fact rows). Stored exactly as indexed, **not normalised** (norms 27-30). Ordered by the original chunk id. |
| `queries` | 60 x 1024 | the 60 benchmark questions embedded with `intfloat/multilingual-e5-large` and the `query: ` prefix (fastembed 0.8.0, onnxruntime 1.30.0) |
| `base_ids` | 1069 | original chunk ids (hex strings), only for traceability |

No text is included. The 10,000 and 50,000-vector datasets are generated from `base` by `bench/data.py:make_vectors`
(seed 7: sample base rows, add N(0, 0.02) noise per dimension, keep the 1,069 originals first, L2-normalise) - the
exact generator of the original run, so results are comparable run to run. Replicating 1,069 vectors ~47x yields tight
clusters of near-duplicates, which is hard for any ANN index: compare the engines with each other, not with public
ANN benchmarks. `scale_vector.py --synthetic` swaps in isotropic random vectors if you prefer data-free input.

`provenance/export_workload_vectors.py` documents how the file was produced (it needs the original project and is
not required to run anything here).
