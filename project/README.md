## PM+CSV segmentation in C++

This is a quick implementation of Perona-Malik + Chan-Sandberg-Vese segmentation in C++.

The idea is to reduce noise in the image with Perona-Malik before segmenting the region with Chan-Sandberg-Vese algorithm, as the latter is relatively sensitive to noise.
The resulting contour is used to cut out ROI from the original image.

### Details

Implementation relies on (version number the code was tested with)
- OpenCV (2.4.8)
- Boost libraries (1.59.0)
- OpenMP (4.0)

The compiler must be compatible with the latest C++14 standard (clang 3.6+ or gcc 5.0+ will do ok).

### Results

Original image (top left), smoothed with Perona-Malik (top right), segmented with Chan-Sandberg-Vese (bottom left), PM+CSV (bottom right).

![seastar_united](https://cloud.githubusercontent.com/assets/6233872/11458132/cf2ec240-96c2-11e5-872b-973bf82380d3.png)

Image courtesy: Wikimedia Commons.

Command used:
```
bin/chan_vese -i seastar.png  -s -N 70 -S -L 0.25 -T 100 -K 30
```
---

Zero level set evolution and the final result, obtained with the following command:
```
bin/chan_vese -i 640px-Europe_night.png \
-N 132 --dt 0.001 -t 0.000001 --nu -293 --lambda1 1 1 0.1 \
-S -L 0.1 -T 1.5 -K 1000 \
-s -V -f 12 -l red
```

![anim](https://cloud.githubusercontent.com/assets/6233872/11458143/12ad0ba8-96c3-11e5-822b-84a0d0492375.gif)

![640px-europe_night_selection](https://cloud.githubusercontent.com/assets/6233872/11458138/f492b67c-96c2-11e5-95f2-342747aff294.png)

Image courtesy: Wikimedia Commons.