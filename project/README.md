## PM+CSV segmentation in C++

This is a quick implementation of Perona-Malik + Chan-Sandberg-Vese segmentation in C++.

The idea is to reduce noise in the image with Perona-Malik before segmenting the region with Chan-Sandberg-Vese algorithm, as the latter is relatively sensitive to noise.
The resulting contour is used to cut out ROI from the original image.

### Details

Implementation relies on (version number the code was tested with)

- OpenCV (2.4.8);
- Boost libraries (1.59.0) (program_options and filesystem libs);
- OpenMP (4.0);

The compiler must be compatible with the latest C++14 standard (clang 3.6+ or gcc 5.0+ will do ok).
To build, just do `make` but make sure that your compiler sees the libs and headers listed above; to read the documentation, do `make doc`; to see all possible command line arguments, do `bin/chan_vese -h`.

### Theory

Both methods summed up in a couple of sentences:

- Perona-Malik segmentation is an improvement from classical Gaussian blur, the kernel of which is a solution to heat equation. Perona and Malik improved upon it by promoting the diffusion constant *c* in the heat equation to a function of the image *I* gradient magnitude aka edge detection function <sup>[1](#perona_malik)</sup>:
![perona_malik](https://cloud.githubusercontent.com/assets/6233872/11458912/d8760df8-96d2-11e5-9de6-f6cd34680b72.png)
Thus, when a region contains no edges (image gradient small), it will be Gaussian-smoothed; when the edge detection function encounters an edge it will not be smoothed but even enhanced.

- Chan-Sandberg-Vese (or Chan-Vese for a single-channel image) formulates optimal contour in the image by defining a functional dependant on zero-level set *u*
![csv_functional](https://cloud.githubusercontent.com/assets/6233872/11458911/d3856c26-96d2-11e5-9e20-16f043a1dd47.png)
where the 1st term penalizes length of the contour; the 2nd term area enclosed by the contour; the 3rd and 4th terms penalize discrepancy between the intensity averages inside and outside of the contour <sup>[2](#csv)</sup> <sup>[3](#chan_vese)</sup>. The corresponding equation of motion for the zero level set can be solved implicitly (read: fast).

For further information, see the documentation or check the references given below.

### Results

Original image (top left), smoothed with Perona-Malik (top right), segmented with Chan-Sandberg-Vese (bottom left), PM+CSV (bottom right).

![seastar_united](https://cloud.githubusercontent.com/assets/6233872/11458132/cf2ec240-96c2-11e5-872b-973bf82380d3.png)

Image courtesy: Wikimedia Commons ([original](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Eleven-Armed_Sea_Star.jpg/370px-Eleven-Armed_Sea_Star.jpg)).

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

Image courtesy: Wikimedia Commons ([original](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Europe_night.png/640px-Europe_night.png)).

---

An alternative to initializing the level set with a checkerboard pattern seen above is to let users specify either rectangular or circular contour:
![screenshot from 2015-11-29 18 20 56](https://cloud.githubusercontent.com/assets/6233872/11458311/19667b48-96c6-11e5-86c1-ecf890041510.png)

Image courtesy: Wikimedia Commons ([original](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/View_of_Earth_is_based_largely_on_observations_from_MODIS.jpg/320px-View_of_Earth_is_based_largely_on_observations_from_MODIS.jpg))

### References

[<a name="perona_malik">1</a>] [Scale-space and edge detection using anisotropic diffusion](http://dx.doi.org/10.1109/34.56205)
[<a name="csv">2</a>] [Active Contours without Edges for Vector-Valued Images](http://dx.doi.org/10.1006/jvci.1999.0442)
[<a name="chan_vese">3</a>] [Chan-Vese Segmentation](http://dx.doi.org/10.5201/ipol.2012.g-cv)