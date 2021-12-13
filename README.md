
# Rowing Pose Tracker

This WIP tool is a largely experimental Computer Vision based coaching application for rowing. It utilizies the Mediapipe library to track the body throughout the stroke, and gives information on the angles of the athlete throughout the stroke. The global variables at the top in the main function can be modified to the desired angles of the coach. This application has not been packaged in any capacity for use. If you were to run this a conda environment would be recommended. Necessary libraries include mediapipe, opencv, numpy, and shutil.


## Usage/Examples

```python
#main.py

filename = 'my_file_to_evaluate.mp4'
...
...
body_finish_window = [30, 40] #[desired lower bound, desired upper bound]
catch_finish_window = [30, 40] #[desired lower bound, desired upper bound]
shin_catch_window = [95, 85] #[desired lower bound, desired upper bound]
...
...
eval_limit=3 #number of strokes to evaluate before program halts
```


## Screenshots

![App Screenshot](https://i.gyazo.com/f146e1fcc17852d5f3c35c0260ee6647.png)
![App Screenshot](https://i.gyazo.com/dff96ed50bd1fa53f54742b1723a67af.png)

