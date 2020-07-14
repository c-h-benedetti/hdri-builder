rm -f output/gaussian/*.png || true
rm -f output/laplacian/*.png || true
rm -f output/*.png
mv -f *.png __OLD__ 2> /dev/null
rm -f *.tif*
