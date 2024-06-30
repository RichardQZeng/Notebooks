# https://github.com/dac0009/Geospatial-Features-in-R.github.io/blob/main/geospatialR.r
# https://www.wvview.org/os_sa/16_LiDAR_and_Image_Analysis.html

library(terra)
library(sf)
library(raster)
library(tmap)
library(tmaptools)
library(lidR)
library(RStoolbox)

library(future)
plan(multissesion, workers = 8L)

# LiDAR

las_cat <- catalog("I:\\LiDea II and South\\las_files")  # Use catalog instead of readLAScatalog
# projection(las_cat) <- "+proj=utm +zone=17 +ellps=GRS80 +datum=NAD83 +units=m +no_defs "
summary(las_cat)
las_check(las_cat)

# Creating a DTM
opt_chunk_size(las_cat) <- 500
plot(las_cat, chunk_pattern = TRUE)
opt_chunk_buffer(las_cat) <- 20
plot(las_cat, chunk_pattern = TRUE)
summary(las_cat)


opt_output_files(las_cat) <- "I:\\LiDea II and South\\las_files\\dtm/dtm_{XLEFT}_{YBOTTOM}"
dtm <- grid_terrain(las_cat, res = 2, knnidw(k = 10, p = 2), keep_lowest = FALSE)

tm_shape(dtm) +
  tm_raster(style= "quantile", palette=get_brewer_pal("Greys", plot=FALSE)) +
  tm_layout(legend.outside = TRUE)

# Creating Hillshade
slope <- terrain(dtm, opt='slope')
aspect <- terrain(dtm, opt='aspect')
hs <- hillShade(slope, aspect, angle=45, direction=315)

tm_shape(hs)+
  tm_raster(style= "cont", palette=get_brewer_pal("Greys", plot=FALSE))+
  tm_layout(legend.outside = TRUE)

# Creating nDMS
opt_output_files(las_cat) <- "I:\\LiDea II and South\\las_files/norm/norm_{XLEFT}_{YBOTTOM}"
lasnorm <- normalize_height(las_cat, dtm)
opt_output_files(las_cat) <- "I:\\LiDea II and South\\las_files/dsm/dsm_{XLEFT}_{YBOTTOM}"
dsm <- grid_canopy(las_cat, res = 2, pitfree(c(0,2,5,10,15), c(0, 1)))

opt_output_files(las_cat) <- "I:\\LiDea II and South\\las_files/chm/chm_{XLEFT}_{YBOTTOM}"
chm <- rasterize_canopy(las_cat, 1, p2r())
ndsm <- dsm - dtm
ndsm[ndsm<0]=0
ndsm
tm_shape(ndsm)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("Greens", n=7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

# Calculate Point Cloud Statistics in Cells
opt_output_files(las_cat) <- "I:\\LiDea II and South\\las_files/means/means_{XLEFT}_{YBOTTOM}"
opt_filter(las_cat) <- "-keep_first"
metrics <- grid_metrics(las_cat, ~mean(Z), 10)

metrics[metrics<0]=0
tm_shape(metrics)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("Greens", n=7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

# Visualize Return Intensity
opt_output_files(las_cat) <- "I:\\LiDea II and South\\las_files/int/int_{XLEFT}_{YBOTTOM}"
opt_filter(las_cat) <- "-keep_first"
int <- grid_metrics(las_cat, ~mean(Intensity), 5)

int[int<0]=0
tm_shape(int)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("-Greys", n=7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

las1 <- readLAS("I:\LiDea II and South\las_files/tiles_338000_5238000_1.laz")
las1_dtm <- grid_terrain(las1, res = 2, knnidw(k = 10, p = 2), keep_lowest = FALSE)
las1_n <- normalize_height(las1, las1_dtm)
las1_vox <- grid_metrics(las1_n, ~sd(Z), res = 5)

# Image Processing

ls8 <- brick("E:/R Projects/spatial/lidar/ls8example.tif")
plotRGB(ls8, r=5, g=4, b=3, stretch="lin")

ndvi <- (ls8$Layer_5-ls8$Layer_4)/((ls8$Layer_5+ls8$Layer_4)+.001)
tm_shape(ndvi)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("Greens", n = 7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

# Check and handle missing values for PCA 
ls8 <- stack("D:/R Projects/spatial/lidar/ls8example.tif")
ls8 <- calc(ls8, function(x) ifelse(is.finite(x), x, NA))

# Principle Compenent Analysis
# Check and handle missing values for PCA 
ls8 <- stack("D:/R Projects/spatial/lidar/ls8example.tif")
ls8 <- calc(ls8, function(x) ifelse(is.finite(x), x, NA))
ls8_pca <- rasterPCA(ls8, nSamples = NULL, nComp = nlayers(ls8), spca = FALSE)

ls8_pca_img <- stack(ls8_pca$map)
plotRGB(ls8_pca_img, r=1, b=2, g=3, stretch="lin")

ls8_pca$model

ls8_pca$model$loadings

pre <- brick("D:/R Projects/spatial/lidar/pre_ref.img")
post <- brick("D:/R Projects/spatial/lidar/post_ref.img")

plotRGB(pre, r=6, g=4, b=2, stretch="lin")
plotRGB(post, r=6, g=4, b=2, stretch="lin")


names(pre) <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
names(post) <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")

pre_brightness <- (pre$Blue*.3561) + (pre$Green*.3972) + (pre$Red*.3904) + (pre$NIR*.6966) + (pre$SWIR1*.2286) + (pre$SWIR2*.1596)
pre_greenness <- (pre$Blue*-.3344) + (pre$Green*-.3544) + (pre$Red*-.4556) + (pre$NIR*.6966) + (pre$SWIR1*-.0242) + (pre$SWIR2*-.2630)
pre_wetness <- (pre$Blue*.2626) + (pre$Green*.2141) + (pre$Red*.0926) + (pre$NIR*.0656) + (pre$SWIR1*-.7629) + (pre$SWIR2*-.5388)
post_brightness <- (post$Blue*.3561) + (post$Green*.3972) + (post$Red*.3904) + (post$NIR*.6966) + (post$SWIR1*.2286) + (post$SWIR2*.1596)
post_greenness <- (post$Blue*-.3344) + (post$Green*-.3544) + (post$Red*-.4556) + (post$NIR*.6966) + (post$SWIR1*-.0242) + (post$SWIR2*-.2630)
post_wetness <- (post$Blue*.2626) + (post$Green*.2141) + (post$Red*.0926) + (post$NIR*.0656) + (post$SWIR1*-.7629) + (post$SWIR2*-.5388)
pre_tc <- stack(pre_brightness, pre_greenness, pre_wetness)
post_tc <- stack(post_brightness, post_greenness, post_wetness)

plotRGB(pre_tc, r=3, g=2, b=1, stretch="lin")
plotRGB(post_tc, r=3, g=2, b=1, stretch="lin")

# Differenced Normalized Burn Ratio (dNBR)

pre_nbr <- (pre$NIR - pre$SWIR2)/((pre$NIR + pre$SWIR2)+.0001)
post_nbr <- (post$NIR - post$SWIR2)/((post$NIR + post$SWIR2)+.0001)
dnbr <- pre_nbr - post_nbr

dnbr[dnbr <= 0] <- NA
tm_shape(dnbr)+
  tm_raster(style= "equal", n=7, palette=get_brewer_pal("YlOrRd", n = 7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

# Moving Windows
ndvi5 <- focal(ndvi, w=matrix(1/25,nrow=5,ncol=5)) 

tm_shape(ndvi5)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("Greens", n = 7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

gx <- c(2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -1, -2, -4, -2, -2) 
gy <- c(2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2)
gx_m <- matrix(gx, nrow=5, ncol=5, byrow=TRUE)
gx_m
gy_m <- matrix(gy, nrow=5, ncol=5, byrow=TRUE)
gy_m
ndvi_edgex <- focal(ndvi, w=gx_m)
ndvi_edgey <- focal(ndvi, w=gy_m) 

tm_shape(ndvi_edgex)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("-Greys", n = 7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)

tm_shape(ndvi_edgey)+
  tm_raster(style= "quantile", n=7, palette=get_brewer_pal("-Greys", n = 7, plot=FALSE))+
  tm_layout(legend.outside = TRUE)