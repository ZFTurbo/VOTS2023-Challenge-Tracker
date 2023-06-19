# VOT 2023 Challenge Trackers
Repository which contains trackers for [VOTS2023 Challenge](https://www.votchallenge.net/vots2023/). The main task is to track given target through the full video. Initial target is given on first frame only as mask. Track is the set of masks throught the whole video. See the example below (_green - groundtruth, red - predicted, yellow - intersection_).  

![giraffe-15](images/example.gif)

Our method depends on 2 models. First model is [Segment Anything](https://github.com/facebookresearch/segment-anything) by Facebook based on VIT-H backbone. This model searches for the regions of interest and returns a set of masks. Next, [Open CLIP model](https://github.com/mlfoundations/open_clip) is used. With this model, we find vectors for each region of interest, as well as for all objects to be found. After that, cosine similarities between each proposed mask and each object are found. Then mask with maximum value of metric is chosen for the object. Tracker searches for all objects at once. We tried to keep tracker as simple as possible, with minimum heuristics. The
tracker is zero-shot. This mean we didn’t train it on any tracking dataset. 

We created 2 trackers:
* [ZFTurbo_HSE_IPPM_tracker_SegmentAnything_and_CLIP.py](ZFTurbo_HSE_IPPM_tracker_SegmentAnything_and_CLIP.py) - main tracker.
* [ZFTurbo_HSE_IPPM_tracker_SegmentAnything_and_Dinov2.py](ZFTurbo_HSE_IPPM_tracker_SegmentAnything_and_Dinov2.py) - OpenCLIP is replaced with [Dinov2](https://github.com/facebookresearch/dinov2).

## Usage

For the usage please refer to [VOT Challenge Toolkit overview](https://www.votchallenge.net/howto/overview.html)

```
vot evaluate --workspace <workspace-path> <tracker-name>
```

## Quality comparison

VOT challenge contained 4 sequences for validation and 144 videos on test set. [More info](https://www.votchallenge.net/vots2023/participation.html). It uses multiple metrics. Description of metrics can be found [here](https://data.votchallenge.net/vots2023/measures.pdf).

### Validation results

| Tracker     | Quality  | Accuracy  | Robustness  | NRE  | DRE  | ADQ |
| ------------- |:---------:|:----------:|:----------:|:----------:|:------------------:|:------------------:|
| SegmentAnything and CLIP   | 0.332   | 0.671    | 0.510     | 0.002    | 0.488  | 0.000  |
| SegmentAnything and Dinov2 | 0.326   | 0.663    | 0.520     |  0.002   | 0.478  | 0.000  |

### Test results

| Tracker     | Quality  | Accuracy  | Robustness  | NRE  | DRE  | ADQ |
| ------------- |:---------:|:----------:|:----------:|:----------:|:------------------:|:------------------:|
| SegmentAnything and CLIP   | 0.25   | 0.66    | 0.37     | 0.01    | 0.62  | 0.000  |


## Additional code

We also created easy visualizer of tracker predictions. It can be useful to check results by eyes.
* Visualize validation data: [utils/visualize_valid.py](utils/visualize_valid.py).
* Visualize test data: [utils/visualize_test.py](utils/visualize_test.py).

### Usage

```bash
python3 utils/visualize_test.py --tracker_name SRZLT_HSE_IPPM_ClipSegmentAnything --workspace /home/vot_workspace
```

## Known issues

* Tracker predicts the required object for each frame. If the required object leaves the frame,
the tracker will predict another random object.
* Tracker always uses only a single template given (from first frame to find the object on each
other frame).
* Tracker doesn’t fix obvious failed cases (for example, if object on the next frame is too far
away from the previous tracked position).
* If there are several similar objects in video, then tracker works bad.

## Citation

TBD
