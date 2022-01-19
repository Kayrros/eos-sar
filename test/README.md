
```
# should only be done once per roi:
python fetch_dem.py rois/iw2_bid5.json

# add --shadows to compute shadows
python terrain_flattening.py rois/iw2_bid5.json sim_iw2_bid5.tiff

python get_corresponding_image.py rois/iw2_bid5.json sim_iw2_bid5.tiff withrtc_iw2_bid5.tiff withoutrtc_iw2_bid5.tiff
```

