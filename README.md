# Point Cloud vs Mesh Graphs for Intraoral Scan Segmentation

Binary per-vertex segmentation (tooth vs. gingiva) of 3D intraoral scans from the Teeth3DS+ benchmark, comparing:
- DGCNN (point cloud): neighbours = k nearest poinst, recomputed per layer
- Mesh GNN: neigbours = vertices connected by mesh edges

Both models share the same EdgeConve architecture, only graph definition differs. Predictions are analysed with Integrated Gradients to see which geometric features (position vs. surface noramls) drive tooth/gingive decisions, especially at the boundary.

*Main results*: mesh connectivity improves k-NN on test mIoU from 0.926 to 0.940, boundary accuracy from 0.774 to 0.798 (3 seeds each), while training is about 3-4x faster with less than half the GPU memory. The mesh model also relies more on local surface shape (~62% of attribution on normals vs. ~51% for DGCNN).

## Background

### Intraoral scans
Intraoral scans capture a 3D model of patient's jaw during a routine dental visit. The result is a triangle mesh of the surface of the teeth and gums.

### Meshes, vertices, faces, normals
- Vertex: a point on the surface, defined by (x,y,z)
- Face: a small traingle connecting three vertices. Thousands of triangles together approximate the curved surface.
- Mesh: all vertices plus all faces. Unlike a plain point cloud, a mesh knows which points are neighbours on the surface.
- Normal: the direction the surface faces at a vertex, it describes local shape

### Task

Every vertex of a scane gets a label: tooth or gingiva. The hardest and clinically most relevants part is the tooth-gingiva boundary, where e.g. gingival recession is measured.

## Method
### Data
- Teeth3DS+ (data part 1), upper jaws only: 151 scans, split into 105 train/ 23 validation/ 23 test (fixed seed, same split for both models).
- Oriringal per-vertex FDI tooth labels are collapsed to binary: label 0 is gingiva, any tooth number is tooth

### Preprocessing (src/preprocess.py)
1. Simplification: quadric decimation to 20,000 traingles (~10,000 vertices per scan), keeping shape detail where curvature is high.
2. Label transfer: each simplified vertex takes the label of the nearest original vertex.
3. Normals: recomputed on the simplified mesh.
4. Normalisation: centered at the origin and scaled into the unit sphere.
5. Mesh graph: every triangle gives edges in both directions, without duplicates.

Input features per vertex: position (3) + normal (3)

### Models (src/models.py)

Both models are the same network, built from EdgeConv layers. For a vertex *i* with neightbours *j*":
<code>
h_i'= max_j MLP([h_i, h_j - h_i])
</code>

The term <code> h_j - h_i </code> captures how the surface change around a vertex. Difference is who counts as a neighbour in model.

||DGCNN|Mesh GNN|
|---|---|---|
|Neighbours| k = 20 nearest points| vertices sharing a mesh edge |
|Graph| dynamic: recomputed in every layer, in the current feature space| fixed: taken from the triangles|
|Strength| can connect similiar regions that are far apart| ifnromation flows only along the surface|
|Weakness| can connect points across small gaps| each layer sees only direct neighbours|

Architecture is identical. It starts with three EdgeConv layer, each producing 64 features per vertex. The first layer works directly on the input features, and each following layer works on the output of the previous one. Every vertex gradually gathers information from a wider neighbourhood. The outputs of all three laters are then joined, giving each vertex 192 features that describe its local surroundings at three different scales. Later the go through a small fully connected layer, and the maximum of each feature over all vertics of the scan in taken. This summary of the entine jaw is then attached to every vertex, next to its own local features. In the end classifier of three fully connected layers turns these combined features into a tooth or gingiva prediction for each vertex.

### Training (src/train.py)

- AdamW (lr 1e-3, weight decay 1e-4), cosine learning-rate schedule, 40 epochs, batch size 2, dropout 0.5
- Augmentation: random rotation (±10° per axis), scaling (0.9–1.1), small jitter
- Model selection: epoch with the best validation mIoU; test set evaluated once at the end
- 3 seeds per model (0, 1, 2); the split is the same for all run

### Metrics
- IoU per class and mIoU, accuracy
- Boundary accuracy: accuracy on vertices that have at least one mesh neighbour of the other class in the ground truth

### Explainability (src/explain.py)

Integrated Gradient (IG) measures how much each input feature of each vertex contributed to a prediction. It integrates the gradient along a straight path from a baseline to the real input; the attributions sum to the difference between the model's score on the input and on the baseline, which is used as a correctness check.

- Explained score: sum over the target vertices of (tooth logit - gingiva logit)
- Targets: regions = all vertices predicted as tooth; boundary = predicted tooth vertices next to a predicted gingiva vertex
- Position vs. normal share: fraction of absolute attribution on the 3 position vs. the 3 normal features
- 256 integration step for all runs
- DGCNN: fixed neighbourhoods. IG assumes the model's output changes smoothly as the input is scaled from zeros to the real scan. DGCNN chooses neirhgtbours in every layer, and in layer 2-3 this choice can switch along the way, which breaks IG. The neirghbours are therefore computed once on the real scan and resused for all IG steps. The attributions explain the prediction for the nerighbourhoods the model actually used, not how the input affects which neighbours are chosen.

## Results
### Segmentation (test set, mean +- std over 3 seeds)
| Model |	Graph |	IoU tooth |	IoU gingiva	mIoU |	Acc |	Boundary acc |
|--------|-----------|---------|------|---------|------|---------|
|DGCNN |	dynamic k-NN |	0.9198 ± 0.0021 |	0.9314 ± 0.0019 |	0.9256 ± 0.0020	|0.9616 ± 0.0011|	0.7742 ± 0.0029|
|Mesh GNN|	mesh edges|	0.9354 ± 0.0022|	0.9444 ± 0.0020	|0.9399 ± 0.0021|	0.9692 ± 0.0011	|0.7983 ± 0.0029|

The mesh GNN is better on every metric and in every seed. The gap in mIoU (+1.4 points) is about 7× the standard deviation; the largest gain is at the gum line (+2.4 points boundary accuracy).

### Efficiency
| Model	 | Parameters |	Time / epoch |	Peak GPU memory |
|DGCNN	|344,834|	~13 s|	2.2 GB|
|Mesh GNN|	344,834	|~3–4.5 s|	0.9 GB|

### Explainability 
Integrated Gradients (256 steps, all-zero baseline) is used. The table shows the mean +- std of the fraction of absolute attribution on position vs. normal features.

| Model | Target | Position | Normals | Completeness error |
|---|---|---|---|---|
| Mesh GNN | region | 39.8 ± 3.5% | 60.2% | < 0.3% on all scans |
| Mesh GNN | boundary | 40.0 ± 3.6% | 60.0% | < 0.6% on all scans |
| DGCNN | region | 49.0 ± 3.8% | 51.0% | < 5% on 14 scans, 5–8% on 3, 17–128% on 4 |
| DGCNN | boundary | 48.8 ± 3.9% | 51.2% | < 3% on all scans (median 0.9%) |

The mesh GNN bases its decisions mainly on **surface shape**, while DGCNN weighs position and shape about equally. The difference is consistent: at the gum line, DGCNN attributes more to position than the mesh GNN on all scans, by at mean 8.8 pp. Both models shift direction from scan to scan, so part of the variation comes from the jaw itself rather than the model.

The larger completeness errors for the DGCNN region target occur on scans where the model's score on the real input and on the baseline nearly cancel, so a small absolute error becomes a large relative one. The boundary results, which are reliable for both model on evert scan, are therefore used as the main result.


## Discussion
- The graph definision matters: With identical architecture and parameter count, mesh connectivity give better segmentation, mostly at the boundary. Since DGCNN also reaches a higher training loss, the difference is not overfitting, the mesh graph is a more useful structure for this task.
- Mesh edges follows the surface, so information does not leak between tooth adn gum across narrow gaps, which k-NN neighbourhoods can do.
- The mesh modle puts more weight on normals, local shape, which is less tied to where a structure happens to lie in the scan and should generalise better to different scanners and jaw poses.

## Limitations
- One data part, binary task only; 
- IG for DGCNN is computed wiht fixed k-NN graphs, it explain the prediction for the neighbourhoods the model used, not the effect of the input on neighbour selection.
- No comparison yet with mesh-specific arhcitectures (e.g. MeshCNN, MeshSegNet)


## Repository structure

<code>
.
├── src/
│   ├── download_data.py   # download Teeth3DS+ parts from OSF (resumable)
│   ├── visualize_scan.py  # 3D viewer for raw (.obj) and preprocessed (.pt) scans
│   ├── preprocess.py      # decimation, label transfer, normals, mesh edges -> .pt + split.json
│   ├── models.py          # shared EdgeConv segmentation model (graph="knn" | "mesh")
│   ├── train.py           # training, validation-based model selection, test metrics
│   └── explain.py         # Integrated Gradients + coloured .ply heatmaps
├── test/                  # pytest tests on synthetic meshes (no dataset needed)
├── results/               # metrics and training histories (JSON)
├── pytest.ini
└── requirements.txt
</code>

## Setup


<code>
python -m venv env
env\Scripts\activate            # Windows (Linux/macOS: source env/bin/activate)
pip install -r requirements.txt
pytest -q                       # runs on CPU with synthetic data

</code>
Install PyTorch with CUDA support first (see pytorch.org). No torch_cluster is needed: the k-NN search is implemented in plain PyTorch.

## Usage

<code>
python src/download_data.py --parts 1             # Teeth3DS+ data part 1 (~1.5 GB)
python src/preprocess.py                          # data/raw -> data/processed (+ split.json)
python src/visualize_scan.py --processed          # inspect a preprocessed scan
python src/train.py --graph mesh                  # mesh GNN
python src/train.py --graph knn                   # DGCNN
python src/explain.py --run runs/mesh --target boundary --steps 256
</code>
The dataset is not redistributed in this repository; see the dataset page for access and terms.
