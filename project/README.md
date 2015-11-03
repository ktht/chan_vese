## Chan-Vese segmentation in C++

This is a quick implementation of Chan-Vese segmentation in C++.

### Details

Implementation relies on (version number the code was tested with)
- OpenCV (2.4.8)
- Boost libraries (1.59.0)
- ncurses (5.9.20140118; UNIX) / PDcurses (untested; Windows)

The compiler must be compatible with the latest C++14 standard (clang 3.6 or gcc 5.0 will do ok).

### TODO

- explain here what Chan-Vese is all about
- add level set reinitialization
- let the user specify the initial contour

### Preliminary results

After 10 iterations with mu=0.1.

![comparison](https://cloud.githubusercontent.com/assets/6233872/10898328/04b6bcfc-81d2-11e5-8672-7974c3fd1366.png)

Image courtesy: Wikimedia Commons.
