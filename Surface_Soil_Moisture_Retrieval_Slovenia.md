```python
import ee
import geemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# Trigger the authentication flow.
ee.Authenticate()
# Initialize the library. (find the name in the GEE code editor under asset tab, written in Blue)
ee.Initialize(project='') # provide username

# Mount drive
from google.colab import drive
drive.mount('') # provide pathway to drive
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    Mounted at /content/drive
    


```python
# Define geemap.Map()
m = geemap.Map()
# Setup country border collection
borders = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
# Select slovenia from country border collection
Slovenia = borders.filterMetadata('country_na', 'equals', 'Slovenia')
AOI = ee.Geometry.BBox(14.4700, 46.0503, 14.4715, 46.0489)
#AOI = ee.Geometry.Point(14.471, 46.049)


```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
border = ee.Image().byte().paint(featureCollection= AOI, color='red', width=2)
m.addLayer(border, {'palette': 'red'}, "border")

m.centerObject(AOI, 16)
m
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    Map(center=[46.04959999959482, 14.470749999864204], controls=(WidgetControl(options=['position', 'transparent_…



```python
# Load and filter Sentinel-1 dataset, see: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
# filter the collection to only include images over slovenia
# select date range
# select type of polarisation (VV, VH, HH OR HV)
# select instument mode and resolution
sentinel_1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
)

m.addLayer(sentinel_1.select('VV').filterDate('2024-01-01', '2024-12-30').mean(), {'palette': ['white', 'black'], 'min': -20, 'max': 5}, 'sentinel')

# Load and filter SMAP dataset, see: https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_007
# filter for slovenia, select upper 5cm of soil, filter date
SMAP = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
    .select('sm_surface')
    .filterDate('2019-01-01', '2024-12-30')
    .map(lambda image: image.clip(AOI))
)

# load Modis NDVI data, see: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1").select('NDVI')
    .filterDate('2015-01-01', '2024-12-30')
)

# Load digital elevation model and calculate slope, see: https://gee-community-catalog.org/projects/fabdem/
FABDEM = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
FABDEM_proj = FABDEM.first().projection();
FABDEM = FABDEM.mosaic().setDefaultProjection(FABDEM_proj).clip(Slovenia)
terrain_slope = ee.Terrain.slope(FABDEM)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Landuse map, see: https://gee-community-catalog.org/projects/glc_fcs/

#Yearly data from 2000-2022
annual = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual') \
            .map(lambda image: image.clip(Slovenia))
#Five-Yearly data for 1985-90, 1990-95 and 1995-2000
fiveyear = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/five-years-map') \
              .map(lambda image: image.clip(Slovenia))

#Classification scheme has 36 classes (35 landcover classes and 1 fill value)
classValues = [10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 122, 130, 140, 150, 152, 153, 181, 182, 183, 184, 185, 186, 187, 190, 200, 201, 202, 210, 220, 0];
classNames = ['Rainfed_cropland', 'Herbaceous_cover_cropland', 'Tree_or_shrub_cover_cropland', 'Irrigated_cropland', 'Open_evergreen_broadleaved_forest', 'Closed_evergreen_broadleaved_forest', 'Open_deciduous_broadleaved_forest', 'Closed_deciduous_broadleaved_forest', 'Open_evergreen_needle_leaved_forest', 'Closed_evergreen_needle_leaved_forest', 'Open_deciduous_needle_leaved_forest', 'Closed_deciduous_needle_leaved_forest', 'Open_mixed_leaf_forest', 'Closed_mixed_leaf_forest', 'Shrubland', 'Evergreen_shrubland', 'Deciduous_shrubland', 'Grassland', 'Lichens_and_mosses', 'Sparse_vegetation', 'Sparse_shrubland', 'Sparse_herbaceous', 'Swamp', 'Marsh', 'Flooded_flat', 'Saline', 'Mangrove', 'Salt_marsh', 'Tidal_flat', 'Impervious_surfaces', 'Bare_areas', 'Consolidated_bare_areas', 'Unconsolidated_bare_areas', 'Water_body', 'Permanent_ice_and_snow', 'Filled_value'];
classColors = ['#ffff64', '#ffff64', '#ffff00', '#aaf0f0', '#4c7300', '#006400', '#a8c800', '#00a000', '#005000', '#003c00', '#286400', '#285000', '#a0b432', '#788200', '#966400', '#964b00', '#966400', '#ffb432', '#ffdcd2', '#ffebaf', '#ffd278', '#ffebaf', '#00a884', '#73ffdf', '#9ebb3b', '#828282', '#f57ab6', '#66cdab', '#444f89', '#c31400', '#fff5d7', '#dcdcdc', '#fff5d7', '#0046c8', '#ffffff', '#ffffff'];

#Mosaic the data into a single image
annualMosaic = annual.mosaic();
fiveYearMosaic = fiveyear.mosaic();

#Rename bands from b1, b2, etc. to 2000, 2001, etc.
fiveYearsList = ee.List.sequence(1985, 1995, 5).map(lambda year: ee.Number(year).format('%04d'))
fiveyearMosaicRenamed = fiveYearMosaic.rename(fiveYearsList)
yearsList = ee.List.sequence(2000, 2022).map(lambda year: ee.Number(year).format('%04d'))
annualMosaicRenamed = annualMosaic.rename(yearsList)
years = fiveYearsList.cat(yearsList)

#Convert the multiband image to an ImageCollection
fiveYearlyMosaics = fiveYearsList.map(lambda year:
    fiveyearMosaicRenamed.select([year]).set({'system:time_start': \
        ee.Date.fromYMD(ee.Number.parse(year), 1, 1).millis(), \
       'system:index': year, 'year': ee.Number.parse(year)}))
yearlyMosaics = yearsList.map(lambda year:
    annualMosaicRenamed.select([year]).set({'system:time_start': \
        ee.Date.fromYMD(ee.Number.parse(year), 1, 1).millis(), \
       'system:index': year, 'year': ee.Number.parse(year)}))
allMosaics = fiveYearlyMosaics.cat(yearlyMosaics)
mosaicsCol = ee.ImageCollection.fromImages(allMosaics)

#Recode the class values, assigning 0 to urban,water and ice
newClassValues = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1]
def renameClasses(image):
    reclassified = image.remap(classValues, newClassValues).rename('classification')
    return reclassified
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Masking backscatter based on slope and landcover

# Landcover mask
year = 2020
landcoverMask = mosaicsCol.map(renameClasses).filter(ee.Filter.eq('year', year)).first()
m.addLayer(landcoverMask, {}, 'landcoverMask')

# Slope mask
slopeMask = (ee.Image(1).clip(Slovenia)
    .where(terrain_slope.lte(20), 1)
    .where(terrain_slope.gt(20), 0)
)
m.addLayer(slopeMask, {}, 'slopeMask')

# Combined landcover & slope mask
def geography_mask(image):
  maskLayer1 = landcoverMask
  maskLayer2 = slopeMask
  doubleMask = landcoverMask.multiply(slopeMask)
  VV = image.select('VV').add(ee.Image(999))
  combined = doubleMask.multiply(VV).rename('masked_VV')
  masked_VV = combined.updateMask(combined.select('masked_VV').neq(0))
  masked_VV = masked_VV.select('masked_VV').subtract(ee.Image(999))
  return image.addBands(masked_VV)

doubleMask = landcoverMask.multiply(slopeMask)
m.addLayer(doubleMask, {}, 'doubleMask')

# Masking backscatter based on energy value
# create mask that removes all images from a collection where -20 < VV < -5
def energy_mask(image):
    VV = image.select('VV')
    masked_VV = VV.updateMask(VV.gt(-20) \
                   .And(VV.lt(-5))).rename('masked_VV')
    return image.addBands(masked_VV)

# Applying no mask
def no_mask(image):
    VV = image.select('VV')
    masked_VV = VV.rename('masked_VV')
    return image.addBands(masked_VV)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Select one masking option:
S1_masked = sentinel_1.map(geography_mask) # Put a '#' in front of the 'S1_masked' you do not want to use
S1_masked = sentinel_1.map(energy_mask) # Put a '#' in front of the 'S1_masked' you do not want to use

S1_geo_masked = sentinel_1.map(geography_mask)
S1_ene_masked = sentinel_1.map(energy_mask)
S1_no_mask = sentinel_1.map(no_mask)

m.addLayer(S1_geo_masked.select('masked_VV').filterDate('2024-01-01', '2024-12-30').mean(), {'palette': ['white', 'black'], 'min': -20, 'max': 5}, 'sentinel_geographymask')
m.addLayer(S1_ene_masked.select('masked_VV').filterDate('2024-01-01', '2024-12-30').mean(), {'palette': ['white', 'black'], 'min': -20, 'max': 5}, 'sentinel_energymask')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# reproject backscatter values to larger cell size
reprojection_scale = 1000 # reprojection_scale value can be changed

# Create frame to which sentinel can be reprojected
frame = ndvi.reduce(ee.Reducer.mean()).rename('frame')
S1_masked = S1_masked.map(lambda image: image.addBands(frame))

S1_geo_masked = S1_geo_masked.map(lambda image: image.addBands(frame))
S1_ene_masked = S1_ene_masked.map(lambda image: image.addBands(frame))
S1_no_mask = S1_no_mask.map(lambda image: image.addBands(frame))

# Reproject function
# reprojects the sentinel-1 data (originally in 10m scale) to a new scale
# new scale is based on previously determined 'reprojection_scale'
def reproject(image):
  masked = image.select("masked_VV").add(ee.Image(999))
  frame = image.select('frame')

  imageReducers = masked.reduceResolution(reducer= ee.Reducer.mean().combine(
      reducer2= ee.Reducer.count(), sharedInputs= True), maxPixels= reprojection_scale*reprojection_scale/100) \
      .reproject(crs= frame.projection(), scale=reprojection_scale)

  mean = imageReducers.select(0).rename('mean')
  count = imageReducers.select(1).rename('count')

  removeHighNullCount = imageReducers.select(0).updateMask(imageReducers.select(1).gte(reprojection_scale/1.5))
  reprojected = removeHighNullCount.subtract(ee.Image(999)).rename("reprojected_VV")

  bands = [mean, count, reprojected]
  return image.addBands(bands)

S1_reprojected = S1_masked.map(reproject)

S1_ene_reprojected = S1_ene_masked.map(reproject)
m.addLayer(S1_ene_reprojected.select('reprojected_VV').filterDate('2024-01-01', '2024-12-30').mean(), {'palette': ['white', 'black'], 'min': -25, 'max': 10}, 'sentinel_ene_reproject')
S1_geo_reprojected = S1_geo_masked.map(reproject)
m.addLayer(S1_geo_reprojected.select('reprojected_VV').filterDate('2024-01-01', '2024-12-30').mean(), {'palette': ['white', 'black'], 'min': -25, 'max': 10}, 'sentinel_geo_reproject')
S1_no_mask_reprojected = S1_no_mask.map(reproject)
m.addLayer(S1_no_mask_reprojected.select('reprojected_VV').filterDate('2024-01-01', '2024-12-30').mean(), {'palette': ['white', 'black'], 'min': -25, 'max': 10}, 'sentinel_no_mask_reproject')

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Turn band data from image collection into pandas data frame

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df.dropna(subset=[*list_of_bands])

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['date'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['longitude', 'latitude', 'date', *list_of_bands]]

    return df
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
S1_reprojected.first().bandNames().getInfo()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    ['VV', 'VH', 'angle', 'masked_VV', 'frame', 'mean', 'count', 'reprojected_VV']




```python
ts1 = S1_reprojected.select('angle', 'reprojected_VV').getRegion(AOI, reprojection_scale/10).getInfo()
ts2 = S1_reprojected.select('masked_VV').getRegion(AOI, reprojection_scale/10).getInfo()

df1 = ee_array_to_df(ts1, ['angle', 'reprojected_VV'])
df2 = ee_array_to_df(ts2, ['masked_VV'])
df = df1.merge(df2)

df.sort_values(by='date')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-618c8c5d-6f28-43ea-b254-e1fc80b6cfac" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date</th>
      <th>angle</th>
      <th>reprojected_VV</th>
      <th>masked_VV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.299095</td>
      <td>-9.182049</td>
      <td>-9.270708</td>
    </tr>
    <tr>
      <th>4939</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.305031</td>
      <td>-9.182049</td>
      <td>-8.269204</td>
    </tr>
    <tr>
      <th>1648</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.303673</td>
      <td>-9.182049</td>
      <td>-8.061343</td>
    </tr>
    <tr>
      <th>3295</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.300457</td>
      <td>-9.182049</td>
      <td>-8.547756</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2014-10-07 05:10:24.590</td>
      <td>39.028278</td>
      <td>-10.108362</td>
      <td>-8.699270</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2025-07-18 16:58:55.000</td>
      <td>42.825893</td>
      <td>-10.319888</td>
      <td>-12.156877</td>
    </tr>
    <tr>
      <th>1647</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.216957</td>
      <td>-10.123288</td>
      <td>-10.935361</td>
    </tr>
    <tr>
      <th>3294</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.221432</td>
      <td>-10.123288</td>
      <td>-11.605867</td>
    </tr>
    <tr>
      <th>4938</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.218010</td>
      <td>-10.123288</td>
      <td>-10.182731</td>
    </tr>
    <tr>
      <th>6555</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.222485</td>
      <td>-10.123288</td>
      <td>-11.628845</td>
    </tr>
  </tbody>
</table>
<p>6556 rows × 6 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-618c8c5d-6f28-43ea-b254-e1fc80b6cfac')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-618c8c5d-6f28-43ea-b254-e1fc80b6cfac button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-618c8c5d-6f28-43ea-b254-e1fc80b6cfac');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-44818862-6fd0-4a82-93ba-1823b5ec2d15">
      <button class="colab-df-quickchart" onclick="quickchart('df-44818862-6fd0-4a82-93ba-1823b5ec2d15')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-44818862-6fd0-4a82-93ba-1823b5ec2d15 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# Adjusting backscatter based on incidence angle
def adjusted_monthly(months):
  df_adjusted_monthly = pd.DataFrame()

  for i in months:
    adjusted_month = df.loc[(df['date'].dt.strftime('%m') == i)]

    X = df['angle'].loc[(df['date'].dt.strftime('%m') == i)]
    Y = df['reprojected_VV'].loc[(df['date'].dt.strftime('%m') == i)]


    slope, intercept = np.polyfit(X, Y, 1)

    adjusted_month['adjusted_VV'] = [(x-slope*(y-40)) for x, y in zip(Y, X)]

    df_adjusted_monthly = pd.concat([df_adjusted_monthly, adjusted_month])
  globals()['df_adjusted_monthly'] = df_adjusted_monthly

adjusted_monthly(['01','02','03','04','05','06','07','08','09','10','11','12'])
df = df_adjusted_monthly.sort_values(by='date')

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
bottom_p = df['adjusted_VV'].quantile([0,.01,.02,.03,.04,.05]).mean()
top_p = df['adjusted_VV'].quantile([1,.99,.98,.97,.96,.95]).mean()


# define variables based on B. Bauer-Marschallinger et al., "Toward Global Soil Moisture Monitoring With Sentinel-1: Harnessing Assets and Overcoming Obstacles," doi: 10.1109/TGRS.2018.2858004
k = (95-5)/(top_p - bottom_p)
d = (95-(k*top_p))
bs_dry = ((0-d)/k)
bs_wet = ((100-d)/k)

df['SSM'] = ((df['adjusted_VV']-bs_dry)/(bs_wet-bs_dry))*(100)

df['SSM'] = df['SSM'][(df['SSM'] < 120)&(df['SSM'] > -20)]
df['SSM'] = df['SSM'].clip(upper=100, lower=0)

df
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-2f8e9bbe-4a1f-4605-bc56-17baccc303f9" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date</th>
      <th>angle</th>
      <th>reprojected_VV</th>
      <th>masked_VV</th>
      <th>adjusted_VV</th>
      <th>SSM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3295</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.300457</td>
      <td>-9.182049</td>
      <td>-8.547756</td>
      <td>-9.645390</td>
      <td>76.107491</td>
    </tr>
    <tr>
      <th>4939</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.305031</td>
      <td>-9.182049</td>
      <td>-8.269204</td>
      <td>-9.645073</td>
      <td>76.117955</td>
    </tr>
    <tr>
      <th>0</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.299095</td>
      <td>-9.182049</td>
      <td>-9.270708</td>
      <td>-9.645484</td>
      <td>76.104376</td>
    </tr>
    <tr>
      <th>1648</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2014-10-03 16:50:04.500</td>
      <td>33.303673</td>
      <td>-9.182049</td>
      <td>-8.061343</td>
      <td>-9.645167</td>
      <td>76.114848</td>
    </tr>
    <tr>
      <th>3296</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2014-10-07 05:10:24.590</td>
      <td>39.029316</td>
      <td>-10.108362</td>
      <td>-6.626195</td>
      <td>-10.175494</td>
      <td>58.572691</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2710</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2025-07-18 16:58:55.000</td>
      <td>42.829594</td>
      <td>-10.319888</td>
      <td>-12.156877</td>
      <td>-10.149849</td>
      <td>59.421002</td>
    </tr>
    <tr>
      <th>1647</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.216957</td>
      <td>-10.123288</td>
      <td>-10.935361</td>
      <td>-10.530902</td>
      <td>46.816501</td>
    </tr>
    <tr>
      <th>6555</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.222485</td>
      <td>-10.123288</td>
      <td>-11.628845</td>
      <td>-10.530570</td>
      <td>46.827488</td>
    </tr>
    <tr>
      <th>3294</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.221432</td>
      <td>-10.123288</td>
      <td>-11.605867</td>
      <td>-10.530633</td>
      <td>46.825395</td>
    </tr>
    <tr>
      <th>4938</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2025-07-19 16:49:36.000</td>
      <td>33.218010</td>
      <td>-10.123288</td>
      <td>-10.182731</td>
      <td>-10.530839</td>
      <td>46.818594</td>
    </tr>
  </tbody>
</table>
<p>6556 rows × 8 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2f8e9bbe-4a1f-4605-bc56-17baccc303f9')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2f8e9bbe-4a1f-4605-bc56-17baccc303f9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2f8e9bbe-4a1f-4605-bc56-17baccc303f9');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-dd3b8793-4e1c-46a4-82b2-2de8317041ab">
      <button class="colab-df-quickchart" onclick="quickchart('df-dd3b8793-4e1c-46a4-82b2-2de8317041ab')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-dd3b8793-4e1c-46a4-82b2-2de8317041ab button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_4376f8b8-a4d4-4c24-bba6-631c4d48cc09">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_4376f8b8-a4d4-4c24-bba6-631c4d48cc09 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Import in-field soil moisture data and turn into dataframe

df_crns = pd.read_excel('') # read file path
columns_titles = ['Date','SWC_CRNS','AVG_of_three_sensors_10cm']
df_crns = df_crns.reindex(columns=columns_titles)
df_crns.columns = ['date','SWC_CRNS','AVG_of_three_sensors_10cm']
df_crns = df_crns.loc[(df_crns['date'] < '2024-12-31')]
df_crns
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-8d94f0b8-2890-45c6-9c6d-f30da0d0c329" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>SWC_CRNS</th>
      <th>AVG_of_three_sensors_10cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-03-05 13:00:00</td>
      <td>44.790060</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-03-05 14:00:00</td>
      <td>41.016166</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03-05 15:00:00</td>
      <td>42.888188</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-03-05 16:00:00</td>
      <td>45.566383</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-03-05 17:00:00</td>
      <td>44.417724</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7190</th>
      <td>2024-12-30 19:00:00</td>
      <td>46.624970</td>
      <td>49.394225</td>
    </tr>
    <tr>
      <th>7191</th>
      <td>2024-12-30 20:00:00</td>
      <td>46.324942</td>
      <td>49.413458</td>
    </tr>
    <tr>
      <th>7192</th>
      <td>2024-12-30 21:00:00</td>
      <td>46.281657</td>
      <td>49.415548</td>
    </tr>
    <tr>
      <th>7193</th>
      <td>2024-12-30 22:00:00</td>
      <td>45.764171</td>
      <td>49.429176</td>
    </tr>
    <tr>
      <th>7194</th>
      <td>2024-12-30 23:00:00</td>
      <td>45.501462</td>
      <td>49.457750</td>
    </tr>
  </tbody>
</table>
<p>7195 rows × 3 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8d94f0b8-2890-45c6-9c6d-f30da0d0c329')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8d94f0b8-2890-45c6-9c6d-f30da0d0c329 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8d94f0b8-2890-45c6-9c6d-f30da0d0c329');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-8fbfeac0-27e8-4d41-857e-6f555382ebe7">
      <button class="colab-df-quickchart" onclick="quickchart('df-8fbfeac0-27e8-4d41-857e-6f555382ebe7')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-8fbfeac0-27e8-4d41-857e-6f555382ebe7 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_470a2370-03c3-49d7-bc79-082d7d4f82f8">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_crns')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_470a2370-03c3-49d7-bc79-082d7d4f82f8 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_crns');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
def adjusted_monthly_SSM(months):
  df_adjusted_monthly = pd.DataFrame()
  # get min and max SSM values per month and their slope in correspondence with point sensor or crns
  for i in months:
    adjusted_month = df.loc[(df['date'].dt.strftime('%m') == i)]
    adjusted_month_crns = df_crns.loc[(df_crns['date'].dt.strftime('%m') == i)]

    adjusted_month['SSM_monthly_min'] = adjusted_month['SSM'].min()
    adjusted_month['SSM_monthly_max'] = adjusted_month['SSM'].max()
    adjusted_month['SSM_monthly_mean'] = adjusted_month['SSM'].mean()
    adjusted_month['sensor_slope_monthly'] = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min()) \
                    /((adjusted_month['SSM_monthly_max']-adjusted_month['SSM_monthly_min']))
    adjusted_month['crns_slope_monthly'] = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min()) \
                    /((adjusted_month['SSM_monthly_max']-adjusted_month['SSM_monthly_min']))


    #adjusted_month['SSM_monthly'] = ((X-top_p)/(bottom_p-top_p))*(100)

    df_adjusted_monthly = pd.concat([df_adjusted_monthly, adjusted_month])
  globals()['df_adjusted_monthly'] = df_adjusted_monthly

adjusted_monthly_SSM(['01','02','03','04','05','06','07','08','09','10','11','12'])
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
df = df_adjusted_monthly.sort_values(by='date')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
df_filtered = df.loc[(df['date'] > pd.to_datetime(df_crns['date'].min()))&(df['date'] < pd.to_datetime(df_crns['date'].max()))]
df_crns = df_crns.loc[(df_crns['date'] < pd.to_datetime(df['date'].max()))]
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# rescaling previously determined SSM data to in-field measurement range

slope_sensors = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min())/(df['adjusted_VV'].max()-df['adjusted_VV'].min())
slope_sensors_filtered = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min())/(df_filtered['adjusted_VV'].max()-df_filtered['adjusted_VV'].min())
slope_sensorsssm = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min())/(df['SSM'].max()-df['SSM'].min())
slope_sensorsssm_filtered = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min())/(df_filtered['SSM'].max()-df_filtered['SSM'].min())
slope_sensors_monthly = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min())/(df['SSM_monthly'].max()-df['SSM_monthly'].min())
slope_sensors_monthly_filtered = (df_crns['AVG_of_three_sensors_10cm'].max()-df_crns['AVG_of_three_sensors_10cm'].min())/(df_filtered['SSM_monthly'].max()-df_filtered['SSM_monthly'].min())

slope_of_change = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min())/(df['adjusted_VV'].max()-df['adjusted_VV'].min())
slope_of_change_filtered = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min())/(df_filtered['adjusted_VV'].max()-df_filtered['adjusted_VV'].min())
slope_ssm = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min())/(df['SSM'].max()-df['SSM'].min())
slope_ssm_filtered = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min())/(df_filtered['SSM'].max()-df_filtered['SSM'].min())
slope_monthly = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min())/(df['SSM_monthly'].max()-df['SSM_monthly'].min())
slope_monthly_filtered = (df_crns['SWC_CRNS'].max()-df_crns['SWC_CRNS'].min())/(df_filtered['SSM_monthly'].max()-df_filtered['SSM_monthly'].min())

component2 = df_crns['SWC_CRNS']
df_filtered['SSM2'] = (abs(df_filtered['adjusted_VV']-df_filtered['adjusted_VV'].min())*slope_of_change_filtered)+component2.min()
df_filtered['SSM22'] = (abs(df_filtered['adjusted_VV']-df_filtered['adjusted_VV'].min())*slope_of_change)+component2.min()
df_filtered['SSM23'] = (abs(df_filtered['SSM']-df_filtered['SSM'].min())*slope_ssm_filtered)+component2.min()
df_filtered['SSM24'] = (abs(df_filtered['SSM']-df_filtered['SSM'].min())*slope_ssm)+component2.min()

df_filtered['SSM210'] = [((abs(x-y)*slope_ssm)+component2.min()) for x, y in zip(df_filtered['SSM'], df_filtered['SSM_monthly_min'])]
df_filtered['SSM211'] = [((abs(x-y)*slope_ssm_filtered)+component2.min()) for x, y in zip(df_filtered['SSM'], df_filtered['SSM_monthly_min'])]
df_filtered['SSM215'] = [((abs(x-y)*z)+component2.min()) for x, y, z in zip(df_filtered['SSM'], df_filtered['SSM_monthly_min'], df_filtered['crns_slope_monthly'])]

component3 = df_crns['AVG_of_three_sensors_10cm']
df_filtered['SSM3'] = (abs(df_filtered['adjusted_VV']-df_filtered['adjusted_VV'].min())*slope_sensors_filtered)+component3.min()
df_filtered['SSM32'] = (abs(df_filtered['adjusted_VV']-df_filtered['adjusted_VV'].min())*slope_sensors)+component3.min()
df_filtered['SSM33'] = (abs(df_filtered['SSM']-df_filtered['SSM'].min())*slope_sensorsssm_filtered)+component3.min()
df_filtered['SSM34'] = (abs(df_filtered['SSM']-df_filtered['SSM'].min())*slope_sensorsssm)+component3.min()

df_filtered['SSM310'] = [((abs(x-y)*slope_sensorsssm)+component3.min()) for x, y in zip(df_filtered['SSM'], df_filtered['SSM_monthly_min'])]
df_filtered['SSM311'] = [((abs(x-y)*slope_sensorsssm_filtered)+component3.min()) for x, y in zip(df_filtered['SSM'], df_filtered['SSM_monthly_min'])]
df_filtered['SSM315'] = [((abs(x-y)*z)+component3.min()) for x, y, z in zip(df_filtered['SSM'], df_filtered['SSM_monthly_min'], df_filtered['sensor_slope_monthly'])]

df_filtered

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-5de47676-faf1-40c8-83f0-441f604447c8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date</th>
      <th>angle</th>
      <th>reprojected_VV</th>
      <th>masked_VV</th>
      <th>adjusted_VV</th>
      <th>SSM</th>
      <th>SSM_monthly_min</th>
      <th>SSM_monthly_max</th>
      <th>...</th>
      <th>SSM210</th>
      <th>SSM211</th>
      <th>SSM215</th>
      <th>SSM3</th>
      <th>SSM32</th>
      <th>SSM33</th>
      <th>SSM34</th>
      <th>SSM310</th>
      <th>SSM311</th>
      <th>SSM315</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2587</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.149033</td>
      <td>-8.918085</td>
      <td>-5.587287</td>
      <td>-9.675606</td>
      <td>75.108012</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>56.357774</td>
      <td>74.268513</td>
      <td>56.357774</td>
      <td>46.097835</td>
      <td>35.128749</td>
      <td>46.967237</td>
      <td>39.245670</td>
      <td>52.269065</td>
      <td>68.266574</td>
      <td>52.269065</td>
    </tr>
    <tr>
      <th>5863</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.150028</td>
      <td>-8.918085</td>
      <td>-7.377108</td>
      <td>-9.675496</td>
      <td>75.111654</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>56.359140</td>
      <td>74.270748</td>
      <td>56.359140</td>
      <td>46.099744</td>
      <td>35.129556</td>
      <td>46.969233</td>
      <td>39.246891</td>
      <td>52.270286</td>
      <td>68.268570</td>
      <td>52.270286</td>
    </tr>
    <tr>
      <th>938</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.144569</td>
      <td>-8.918085</td>
      <td>-7.573000</td>
      <td>-9.676099</td>
      <td>75.091688</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>56.351648</td>
      <td>74.258494</td>
      <td>56.351648</td>
      <td>46.089278</td>
      <td>35.125131</td>
      <td>46.958288</td>
      <td>39.240199</td>
      <td>52.263594</td>
      <td>68.257626</td>
      <td>52.263594</td>
    </tr>
    <tr>
      <th>4232</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.145565</td>
      <td>-8.918085</td>
      <td>-8.052552</td>
      <td>-9.675989</td>
      <td>75.095330</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>56.353015</td>
      <td>74.260729</td>
      <td>56.353015</td>
      <td>46.091187</td>
      <td>35.125938</td>
      <td>46.960285</td>
      <td>39.241420</td>
      <td>52.264814</td>
      <td>68.259622</td>
      <td>52.264814</td>
    </tr>
    <tr>
      <th>5864</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-03-12 05:11:11</td>
      <td>38.937450</td>
      <td>-9.165106</td>
      <td>-6.625970</td>
      <td>-9.282593</td>
      <td>88.108081</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>61.236199</td>
      <td>82.247018</td>
      <td>61.236199</td>
      <td>52.912309</td>
      <td>38.009744</td>
      <td>54.093476</td>
      <td>43.602981</td>
      <td>56.626376</td>
      <td>75.392813</td>
      <td>56.626376</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-12-25 05:11:08</td>
      <td>38.935375</td>
      <td>-9.649894</td>
      <td>-9.174757</td>
      <td>-9.747653</td>
      <td>72.724839</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>55.463461</td>
      <td>72.805893</td>
      <td>55.463461</td>
      <td>44.848606</td>
      <td>34.600605</td>
      <td>45.660855</td>
      <td>38.446888</td>
      <td>51.470283</td>
      <td>66.960192</td>
      <td>51.470283</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.827492</td>
      <td>-10.343639</td>
      <td>-10.209711</td>
      <td>-10.084007</td>
      <td>61.598919</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>51.288332</td>
      <td>65.977606</td>
      <td>51.288332</td>
      <td>39.016537</td>
      <td>32.134946</td>
      <td>39.561966</td>
      <td>34.717747</td>
      <td>47.741142</td>
      <td>60.861304</td>
      <td>47.741142</td>
    </tr>
    <tr>
      <th>2659</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.831242</td>
      <td>-10.343639</td>
      <td>-9.220174</td>
      <td>-10.083662</td>
      <td>61.610308</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>51.292606</td>
      <td>65.984596</td>
      <td>51.292606</td>
      <td>39.022507</td>
      <td>32.137471</td>
      <td>39.568210</td>
      <td>34.721565</td>
      <td>47.744960</td>
      <td>60.867547</td>
      <td>47.744960</td>
    </tr>
    <tr>
      <th>5936</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.831982</td>
      <td>-10.343639</td>
      <td>-7.579136</td>
      <td>-10.083594</td>
      <td>61.612556</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>51.293449</td>
      <td>65.985975</td>
      <td>51.293449</td>
      <td>39.023685</td>
      <td>32.137969</td>
      <td>39.569442</td>
      <td>34.722318</td>
      <td>47.745713</td>
      <td>60.868780</td>
      <td>47.745713</td>
    </tr>
    <tr>
      <th>4304</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.828232</td>
      <td>-10.343639</td>
      <td>-9.937670</td>
      <td>-10.083939</td>
      <td>61.601167</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>51.289175</td>
      <td>65.978985</td>
      <td>51.289175</td>
      <td>39.017715</td>
      <td>32.135445</td>
      <td>39.563199</td>
      <td>34.718501</td>
      <td>47.741896</td>
      <td>60.862536</td>
      <td>47.741896</td>
    </tr>
  </tbody>
</table>
<p>294 rows × 28 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5de47676-faf1-40c8-83f0-441f604447c8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5de47676-faf1-40c8-83f0-441f604447c8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5de47676-faf1-40c8-83f0-441f604447c8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-8c2a33f8-4e55-455e-93c8-93220696c09a">
      <button class="colab-df-quickchart" onclick="quickchart('df-8c2a33f8-4e55-455e-93c8-93220696c09a')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-8c2a33f8-4e55-455e-93c8-93220696c09a button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_4a900d90-d7e5-4049-88e6-79aeafebf1c4">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_filtered')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_4a900d90-d7e5-4049-88e6-79aeafebf1c4 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_filtered');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
start = '2024-06-15'
end = '2024-09-10'

df_crns_w = df_crns.loc[((df_crns['date'] >= start) & (df_crns['date'] <= end))]
df_crns_d = pd.concat([df_crns.loc[(df_crns['date'] <= start)], df_crns[(df_crns['date'] >= end)]])

df_filtered_w = df_filtered.loc[((df_filtered['date'] >= start) & (df_filtered['date'] <= end))]
df_filtered_d = pd.concat([df_filtered.loc[(df_filtered['date'] <= start)], df_filtered[(df_filtered['date'] >= end)]])

component2= 'SSM210'
component3= 'SSM310'

slope_w = (df_crns_w['SWC_CRNS'].max()-df_crns_w['SWC_CRNS'].min())/(df_filtered_w[component2].max()-df_filtered_w[component2].min())
slope_d = (df_crns_d['SWC_CRNS'].max()-df_crns_d['SWC_CRNS'].min())/(df_filtered_d[component2].max()-df_filtered_d[component2].min())
slope_w2 = (df_crns_w['SWC_CRNS'].max()-df_crns_w['SWC_CRNS'].min())/(df_filtered_w['SSM'].max()-df_filtered_w['SSM'].min())
slope_d2 = (df_crns_d['SWC_CRNS'].max()-df_crns_d['SWC_CRNS'].min())/(df_filtered_d['SSM'].max()-df_filtered_d['SSM'].min())
slope_sensors_w = (df_crns_w['AVG_of_three_sensors_10cm'].max()-df_crns_w['AVG_of_three_sensors_10cm'].min())/(df_filtered_w[component3].max()-df_filtered_w[component3].min())
slope_sensors_d = (df_crns_d['AVG_of_three_sensors_10cm'].max()-df_crns_d['AVG_of_three_sensors_10cm'].min())/(df_filtered_d[component3].max()-df_filtered_d[component3].min())
slope_sensors_w2 = (df_crns_w['AVG_of_three_sensors_10cm'].max()-df_crns_w['AVG_of_three_sensors_10cm'].min())/(df_filtered_w['SSM'].max()-df_filtered_w['SSM'].min())
slope_sensors_d2 = (df_crns_d['AVG_of_three_sensors_10cm'].max()-df_crns_d['AVG_of_three_sensors_10cm'].min())/(df_filtered_d['SSM'].max()-df_filtered_d['SSM'].min())

df_filtered_w['SSM2_wd'] = (abs(df_filtered_w[component2]-df_filtered_w[component2].min())*slope_w)+df_crns_w['SWC_CRNS'].min()
df_filtered_d['SSM2_wd'] = (abs(df_filtered_d[component2]-df_filtered_d[component2].min())*slope_d)+df_crns_d['SWC_CRNS'].min()
df_filtered_w['SSM3_wd'] = (abs(df_filtered_w[component3]-df_filtered_w[component3].min())*slope_sensors_w)+df_crns_w['AVG_of_three_sensors_10cm'].min()
df_filtered_d['SSM3_wd'] = (abs(df_filtered_d[component3]-df_filtered_d[component3].min())*slope_sensors_d)+df_crns_d['AVG_of_three_sensors_10cm'].min()

df_filtered_w['SSM2_wd2'] = (abs(df_filtered_w['SSM']-df_filtered_w['SSM'].min())*slope_w2)+df_crns_w['SWC_CRNS'].min()
df_filtered_d['SSM2_wd2'] = (abs(df_filtered_d['SSM']-df_filtered_d['SSM'].min())*slope_d2)+df_crns_d['SWC_CRNS'].min()
df_filtered_w['SSM3_wd2'] = (abs(df_filtered_w['SSM']-df_filtered_w['SSM'].min())*slope_sensors_w2)+df_crns_w['AVG_of_three_sensors_10cm'].min()
df_filtered_d['SSM3_wd2'] = (abs(df_filtered_d['SSM']-df_filtered_d['SSM'].min())*slope_sensors_d2)+df_crns_d['AVG_of_three_sensors_10cm'].min()


df_filtered = pd.concat([df_filtered_w, df_filtered_d])
df_filtered.sort_values(by='date')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-f6170bdc-74bb-43a6-a382-741651bdd798" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date</th>
      <th>angle</th>
      <th>reprojected_VV</th>
      <th>masked_VV</th>
      <th>adjusted_VV</th>
      <th>SSM</th>
      <th>SSM_monthly_min</th>
      <th>SSM_monthly_max</th>
      <th>...</th>
      <th>SSM32</th>
      <th>SSM33</th>
      <th>SSM34</th>
      <th>SSM310</th>
      <th>SSM311</th>
      <th>SSM315</th>
      <th>SSM2_wd</th>
      <th>SSM3_wd</th>
      <th>SSM2_wd2</th>
      <th>SSM3_wd2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2587</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.149033</td>
      <td>-8.918085</td>
      <td>-5.587287</td>
      <td>-9.675606</td>
      <td>75.108012</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>35.128749</td>
      <td>46.967237</td>
      <td>39.245670</td>
      <td>52.269065</td>
      <td>68.266574</td>
      <td>52.269065</td>
      <td>59.925957</td>
      <td>56.517991</td>
      <td>54.034420</td>
      <td>52.339524</td>
    </tr>
    <tr>
      <th>5863</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.150028</td>
      <td>-8.918085</td>
      <td>-7.377108</td>
      <td>-9.675496</td>
      <td>75.111654</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>35.129556</td>
      <td>46.969233</td>
      <td>39.246891</td>
      <td>52.270286</td>
      <td>68.268570</td>
      <td>52.270286</td>
      <td>59.927573</td>
      <td>56.519137</td>
      <td>54.036389</td>
      <td>52.340921</td>
    </tr>
    <tr>
      <th>4232</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.145565</td>
      <td>-8.918085</td>
      <td>-8.052552</td>
      <td>-9.675989</td>
      <td>75.095330</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>35.125938</td>
      <td>46.960285</td>
      <td>39.241420</td>
      <td>52.264814</td>
      <td>68.259622</td>
      <td>52.264814</td>
      <td>59.920332</td>
      <td>56.514002</td>
      <td>54.027564</td>
      <td>52.334662</td>
    </tr>
    <tr>
      <th>938</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-03-08 16:50:55</td>
      <td>33.144569</td>
      <td>-8.918085</td>
      <td>-7.573000</td>
      <td>-9.676099</td>
      <td>75.091688</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>35.125131</td>
      <td>46.958288</td>
      <td>39.240199</td>
      <td>52.263594</td>
      <td>68.257626</td>
      <td>52.263594</td>
      <td>59.918717</td>
      <td>56.512856</td>
      <td>54.025596</td>
      <td>52.333266</td>
    </tr>
    <tr>
      <th>2588</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2024-03-12 05:11:11</td>
      <td>38.936123</td>
      <td>-9.165106</td>
      <td>-6.796964</td>
      <td>-9.282740</td>
      <td>88.103226</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>38.008668</td>
      <td>54.090814</td>
      <td>43.601353</td>
      <td>56.624748</td>
      <td>75.390152</td>
      <td>56.624748</td>
      <td>65.690145</td>
      <td>60.606138</td>
      <td>61.059396</td>
      <td>57.321862</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-12-25 05:11:08</td>
      <td>38.935375</td>
      <td>-9.649894</td>
      <td>-9.174757</td>
      <td>-9.747653</td>
      <td>72.724839</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>34.600605</td>
      <td>45.660855</td>
      <td>38.446888</td>
      <td>51.470283</td>
      <td>66.960192</td>
      <td>51.470283</td>
      <td>58.868872</td>
      <td>55.768272</td>
      <td>52.746121</td>
      <td>51.425821</td>
    </tr>
    <tr>
      <th>2659</th>
      <td>14.47141</td>
      <td>46.048989</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.831242</td>
      <td>-10.343639</td>
      <td>-9.220174</td>
      <td>-10.083662</td>
      <td>61.610308</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>32.137471</td>
      <td>39.568210</td>
      <td>34.721565</td>
      <td>47.744960</td>
      <td>60.867547</td>
      <td>47.744960</td>
      <td>53.938885</td>
      <td>52.271767</td>
      <td>46.737808</td>
      <td>47.164533</td>
    </tr>
    <tr>
      <th>5936</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.831982</td>
      <td>-10.343639</td>
      <td>-7.579136</td>
      <td>-10.083594</td>
      <td>61.612556</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>32.137969</td>
      <td>39.569442</td>
      <td>34.722318</td>
      <td>47.745713</td>
      <td>60.868780</td>
      <td>47.745713</td>
      <td>53.939882</td>
      <td>52.272475</td>
      <td>46.739023</td>
      <td>47.165395</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.827492</td>
      <td>-10.343639</td>
      <td>-10.209711</td>
      <td>-10.084007</td>
      <td>61.598919</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>32.134946</td>
      <td>39.561966</td>
      <td>34.717747</td>
      <td>47.741142</td>
      <td>60.861304</td>
      <td>47.741142</td>
      <td>53.933833</td>
      <td>52.268184</td>
      <td>46.731651</td>
      <td>47.160166</td>
    </tr>
    <tr>
      <th>4304</th>
      <td>14.470512</td>
      <td>46.049887</td>
      <td>2024-12-26 16:59:01</td>
      <td>42.828232</td>
      <td>-10.343639</td>
      <td>-9.937670</td>
      <td>-10.083939</td>
      <td>61.601167</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>32.135445</td>
      <td>39.563199</td>
      <td>34.718501</td>
      <td>47.741896</td>
      <td>60.862536</td>
      <td>47.741896</td>
      <td>53.934830</td>
      <td>52.268892</td>
      <td>46.732866</td>
      <td>47.161028</td>
    </tr>
  </tbody>
</table>
<p>294 rows × 32 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f6170bdc-74bb-43a6-a382-741651bdd798')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f6170bdc-74bb-43a6-a382-741651bdd798 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f6170bdc-74bb-43a6-a382-741651bdd798');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f0fec7bf-3858-4220-8be8-b6f7bcb9d94b">
      <button class="colab-df-quickchart" onclick="quickchart('df-f0fec7bf-3858-4220-8be8-b6f7bcb9d94b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f0fec7bf-3858-4220-8be8-b6f7bcb9d94b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# Time series of ndvi based on Landsat9
L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA")

def addNDVI(image):
  nir = image.select('B5')
  red = image.select('B4')
  ndvi = ((nir.subtract(red)).divide(nir.add(red))).rename('NDVI')
  return image.addBands(ndvi)

L9 = L9.filterMetadata('CLOUD_COVER', 'less_Than', 50).map(addNDVI)


ts_l9 = L9.select('NDVI').filterDate(df_crns['date'].min(), df_crns['date'].max()).getRegion(AOI, 10).getInfo()
df_l9 = ee_array_to_df(ts_l9, ['NDVI'])

df_l9.sort_values(by='date')
agg_functions = {'date': 'first', 'NDVI': 'mean'}
df_l9r = df_l9.groupby((df_l9['date'])).aggregate(agg_functions)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Timeseries of ndvi based on MODIS
ts_ndvi = ndvi.select('NDVI').filterDate(df_crns['date'].min(), df_crns['date'].max()).getRegion(AOI, 10).getInfo()
df_ndvi = ee_array_to_df(ts_ndvi, ['NDVI'])

df_ndvi['NDVI'] = df_ndvi['NDVI']*0.0001
df_ndvi.sort_values(by='date')
agg_functions = {'date': 'first', 'NDVI': 'mean'}
df_ndvir = df_ndvi.groupby((df_ndvi['date'])).aggregate(agg_functions)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Time series of ndvi based on Landsat8
L8_TOA = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")

def addNDVI_TOA(image):
  nir = image.select('B5')
  red = image.select('B4')
  ndvi = ((nir.subtract(red)).divide(nir.add(red))).rename('NDVI')
  return image.addBands(ndvi)

L8_TOA = L8_TOA.filterMetadata('CLOUD_COVER', 'less_Than', 50).map(addNDVI_TOA)

ts_l8 = L8_TOA.select('NDVI').filterDate(df_crns['date'].min(), df_crns['date'].max()).getRegion(AOI, 10).getInfo()
df_l8 = ee_array_to_df(ts_l8, ['NDVI'])

df_l8.sort_values(by='date')
agg_functions = {'date': 'first', 'NDVI': 'mean'}
df_l8r = df_l8.groupby((df_l8['date'])).aggregate(agg_functions)

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# combining all NDVI datasets into one
frames = [df_l8r, df_l9r, df_ndvir]
df_merged_ndvi_filtered = pd.concat(frames)

df_merged_ndvi_filtered = df_merged_ndvi_filtered.loc[(df_merged_ndvi_filtered['date'] > pd.to_datetime(df_crns['date'].min()))]

length = len(df_merged_ndvi_filtered)
array = []
for i in range(length):
  array.append(i)

df_merged_ndvi_filtered['index'] = array
df_merged_ndvi_filtered = df_merged_ndvi_filtered.set_index(['index'])

df_merged_ndvi_filtered = df_merged_ndvi_filtered.groupby(pd.Grouper(key='date', freq='96h')).mean().dropna()
df_merged_ndvi_filtered['date'] = df_merged_ndvi_filtered.index

length = len(df_merged_ndvi_filtered)
array = []
for i in range(length):
  array.append(i)

df_merged_ndvi_filtered['index'] = array
df_merged_ndvi_filtered = df_merged_ndvi_filtered.set_index(['index'])
df_merged_ndvi_filtered.describe()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-e333a683-9f28-41db-9ef8-d029cc9e6a21" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDVI</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>29.000000</td>
      <td>29</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.478793</td>
      <td>2024-08-09 17:22:45.517241344</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.058016</td>
      <td>2024-03-19 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.441665</td>
      <td>2024-06-07 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.520139</td>
      <td>2024-08-10 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.595186</td>
      <td>2024-10-21 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.719229</td>
      <td>2024-12-24 00:00:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.169060</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e333a683-9f28-41db-9ef8-d029cc9e6a21')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e333a683-9f28-41db-9ef8-d029cc9e6a21 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e333a683-9f28-41db-9ef8-d029cc9e6a21');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-35ee2cda-8f61-493e-bb41-26e66ca555cf">
      <button class="colab-df-quickchart" onclick="quickchart('df-35ee2cda-8f61-493e-bb41-26e66ca555cf')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-35ee2cda-8f61-493e-bb41-26e66ca555cf button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
m_ndvi = geemap.Map()
m_ndvi.addLayer(L8_TOA.select('NDVI'), {
    'min': 0,
    'max': 1,
    'palette': ['ff0000', 'ffff00', '008000']
}, 'L8_TOA')

m_ndvi.addLayer(L9.select('NDVI'), {
    'min': 0,
    'max': 1,
    'palette': ['ff0000', 'ffff00', '008000']
}, 'L9')

m_ndvi.addLayer(ndvi.select('NDVI').map(lambda image: image.multiply(0.0001)), {
    'min': 0,
    'max': 1,
    'palette': ['ff0000', 'ffff00', '008000']
}, 'ndvi')

m_ndvi.center_object(AOI, 11)
m_ndvi
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    Map(center=[46.04959999959482, 14.470749999864204], controls=(WidgetControl(options=['position', 'transparent_…



```python
# when using premade LAI satellite imagery I get NaN results,
# possibly due to the plot being located within city boundaries
# therefor better to create own LAI based on 'raw' MODIS footage

def SR_LAI(image):

  red = image.select('sur_refl_b01').multiply(0.0001)
  nir = image.select('sur_refl_b02').multiply(0.0001)
  blue = image.select('sur_refl_b03').multiply(0.0001)

  # General formula: 2.5 * (NIR - RED) / ((NIR + 6*RED - 7.5*BLUE) + 1)
  # EVI should be between -1 to 1, for healthy vegetation this is 0.2-0.8
  EVI = ee.Image(2.5).multiply(nir.subtract(red)) \
  .divide((nir.add((ee.Image(6)).multiply(red)).subtract((ee.Image(7.5).multiply(blue)))).add(ee.Image(1)))

  # LAI should be between 0 and 3.5, although values can go much higher
  LAI = (ee.Image(3.618).multiply(ee.Image(EVI)).subtract(ee.Image(0.118))).rename('LAI')
  return image.addBands(LAI)


```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# add LAI band to MODIS collection and turn into dataframe
MOD_LAI = ee.ImageCollection("MODIS/061/MOD09A1") \
  .filterDate(df_filtered['date'].min(), df_filtered['date'].max()).map(SR_LAI)

ts_lai = MOD_LAI.select('LAI').getRegion(AOI, 100).getInfo()

df_lai = ee_array_to_df(ts_lai, ['LAI'])
df_lai
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-31c6fe51-b632-49d1-af7b-59035fd403e0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date</th>
      <th>LAI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-03-13</td>
      <td>0.758055</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-03-21</td>
      <td>0.849716</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-04-06</td>
      <td>1.136194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-04-14</td>
      <td>0.948135</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.470512</td>
      <td>46.048989</td>
      <td>2024-04-22</td>
      <td>1.500949</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>143</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-11-24</td>
      <td>0.672034</td>
    </tr>
    <tr>
      <th>144</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-12-02</td>
      <td>0.618311</td>
    </tr>
    <tr>
      <th>145</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-12-10</td>
      <td>1.001996</td>
    </tr>
    <tr>
      <th>146</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-12-18</td>
      <td>0.542733</td>
    </tr>
    <tr>
      <th>147</th>
      <td>14.47141</td>
      <td>46.049887</td>
      <td>2024-12-26</td>
      <td>0.495463</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 4 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-31c6fe51-b632-49d1-af7b-59035fd403e0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-31c6fe51-b632-49d1-af7b-59035fd403e0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-31c6fe51-b632-49d1-af7b-59035fd403e0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-a13e1f23-916f-4840-91b1-77f7765dc433">
      <button class="colab-df-quickchart" onclick="quickchart('df-a13e1f23-916f-4840-91b1-77f7765dc433')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-a13e1f23-916f-4840-91b1-77f7765dc433 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_060fa142-f66c-4588-88d1-e5a7fdcd69d7">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_lai')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_060fa142-f66c-4588-88d1-e5a7fdcd69d7 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_lai');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Remove duplicate dates from dataframe
# instead of multiple measurements per day, only an average per day will be allowed
agg_functions = {'date': 'first', 'SSM2': 'mean','SSM22': 'mean','SSM23': 'mean','SSM24': 'mean', \
                 'SSM210': 'mean','SSM211': 'mean','SSM215': 'mean', 'SSM':'mean', \
                 'SSM3': 'mean','SSM32': 'mean','SSM33': 'mean','SSM34': 'mean', \
                 'SSM310': 'mean','SSM311': 'mean','SSM315': 'mean',\
                 'SSM2_wd': 'mean','SSM2_wd2': 'mean', 'SSM3_wd': 'mean','SSM3_wd2': 'mean', 'SSM_monthly': 'mean', 'masked_VV':'mean', 'adjusted_VV':'mean'}
df_filtered = df_filtered.groupby((df_filtered['date'])).aggregate(agg_functions)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
length = len(df_filtered)
array = []
for i in range(length):
  array.append(i)

df_filtered['index'] = array
df_filtered = df_filtered.set_index(['index'])
df_filtered
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-81c40231-b078-4287-a302-471880fafc8c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>SSM2</th>
      <th>SSM22</th>
      <th>SSM23</th>
      <th>SSM24</th>
      <th>SSM210</th>
      <th>SSM211</th>
      <th>SSM215</th>
      <th>SSM</th>
      <th>SSM3</th>
      <th>...</th>
      <th>SSM310</th>
      <th>SSM311</th>
      <th>SSM315</th>
      <th>SSM2_wd</th>
      <th>SSM2_wd2</th>
      <th>SSM3_wd</th>
      <th>SSM3_wd2</th>
      <th>SSM_monthly</th>
      <th>masked_VV</th>
      <th>adjusted_VV</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-03-08 16:50:55</td>
      <td>49.444771</td>
      <td>37.165979</td>
      <td>50.417979</td>
      <td>41.774460</td>
      <td>56.355394</td>
      <td>74.264621</td>
      <td>56.355394</td>
      <td>75.101671</td>
      <td>46.094511</td>
      <td>...</td>
      <td>52.266940</td>
      <td>68.263098</td>
      <td>52.266940</td>
      <td>59.923145</td>
      <td>54.030992</td>
      <td>56.515996</td>
      <td>52.337093</td>
      <td>75.296743</td>
      <td>-7.147487</td>
      <td>-9.675797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-03-12 05:11:11</td>
      <td>57.080811</td>
      <td>40.394313</td>
      <td>58.403370</td>
      <td>46.657096</td>
      <td>61.238030</td>
      <td>82.250013</td>
      <td>61.238030</td>
      <td>88.112961</td>
      <td>52.914867</td>
      <td>...</td>
      <td>56.628011</td>
      <td>75.395488</td>
      <td>56.628011</td>
      <td>65.694463</td>
      <td>61.064658</td>
      <td>60.609200</td>
      <td>57.325594</td>
      <td>89.558176</td>
      <td>-7.415497</td>
      <td>-9.282446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-03-13 16:59:05</td>
      <td>56.556360</td>
      <td>40.172588</td>
      <td>57.854926</td>
      <td>46.321751</td>
      <td>60.902686</td>
      <td>81.701568</td>
      <td>60.902686</td>
      <td>87.219332</td>
      <td>52.446438</td>
      <td>...</td>
      <td>56.328488</td>
      <td>74.905628</td>
      <td>56.328488</td>
      <td>65.298083</td>
      <td>60.581579</td>
      <td>60.328075</td>
      <td>56.982979</td>
      <td>88.578686</td>
      <td>-10.013273</td>
      <td>-9.309461</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-03-20 16:50:55</td>
      <td>48.485223</td>
      <td>36.760306</td>
      <td>49.414532</td>
      <td>41.160906</td>
      <td>55.741841</td>
      <td>73.261174</td>
      <td>55.741841</td>
      <td>73.466668</td>
      <td>45.237463</td>
      <td>...</td>
      <td>51.718926</td>
      <td>67.366840</td>
      <td>51.718926</td>
      <td>59.197919</td>
      <td>53.147140</td>
      <td>56.001643</td>
      <td>51.710236</td>
      <td>73.504646</td>
      <td>-9.735648</td>
      <td>-9.725226</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-03-24 05:10:55</td>
      <td>53.089312</td>
      <td>38.706804</td>
      <td>54.229259</td>
      <td>44.104852</td>
      <td>58.685786</td>
      <td>78.075901</td>
      <td>58.685786</td>
      <td>81.311720</td>
      <td>49.349741</td>
      <td>...</td>
      <td>54.348399</td>
      <td>71.667256</td>
      <td>54.348399</td>
      <td>62.677688</td>
      <td>57.388031</td>
      <td>58.469607</td>
      <td>54.718013</td>
      <td>82.103462</td>
      <td>-8.671692</td>
      <td>-9.488058</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2024-12-13 05:11:09</td>
      <td>43.417975</td>
      <td>34.617995</td>
      <td>44.115454</td>
      <td>37.920807</td>
      <td>52.501741</td>
      <td>67.962097</td>
      <td>52.501741</td>
      <td>64.832422</td>
      <td>40.711499</td>
      <td>...</td>
      <td>48.824935</td>
      <td>62.633811</td>
      <td>48.824935</td>
      <td>55.368093</td>
      <td>48.479624</td>
      <td>53.285408</td>
      <td>48.399885</td>
      <td>67.623528</td>
      <td>-7.962102</td>
      <td>-9.986253</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2024-12-14 16:59:03</td>
      <td>38.309224</td>
      <td>32.458138</td>
      <td>38.772977</td>
      <td>34.654170</td>
      <td>49.235104</td>
      <td>62.619619</td>
      <td>49.235104</td>
      <td>56.127460</td>
      <td>36.148466</td>
      <td>...</td>
      <td>45.907241</td>
      <td>57.862018</td>
      <td>45.907241</td>
      <td>51.506900</td>
      <td>43.773881</td>
      <td>50.546926</td>
      <td>45.062421</td>
      <td>59.536812</td>
      <td>-9.832858</td>
      <td>-10.249417</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2024-12-21 16:50:52</td>
      <td>44.072676</td>
      <td>34.894787</td>
      <td>44.800109</td>
      <td>38.339436</td>
      <td>52.920370</td>
      <td>68.646751</td>
      <td>52.920370</td>
      <td>65.947989</td>
      <td>41.296265</td>
      <td>...</td>
      <td>49.198846</td>
      <td>63.245331</td>
      <td>49.198846</td>
      <td>55.862916</td>
      <td>49.082679</td>
      <td>53.636352</td>
      <td>48.827591</td>
      <td>68.659864</td>
      <td>-6.983944</td>
      <td>-9.952528</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2024-12-25 05:11:08</td>
      <td>48.045908</td>
      <td>36.574574</td>
      <td>48.955117</td>
      <td>40.879999</td>
      <td>55.460933</td>
      <td>72.801760</td>
      <td>55.460933</td>
      <td>72.718104</td>
      <td>44.845075</td>
      <td>...</td>
      <td>51.468025</td>
      <td>66.956500</td>
      <td>51.468025</td>
      <td>58.865884</td>
      <td>52.742479</td>
      <td>55.766153</td>
      <td>51.423239</td>
      <td>74.949151</td>
      <td>-8.850171</td>
      <td>-9.747856</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2024-12-26 16:59:01</td>
      <td>41.524304</td>
      <td>33.817397</td>
      <td>42.135148</td>
      <td>36.709956</td>
      <td>51.290891</td>
      <td>65.981790</td>
      <td>51.290891</td>
      <td>61.605738</td>
      <td>39.020111</td>
      <td>...</td>
      <td>47.743428</td>
      <td>60.865042</td>
      <td>47.743428</td>
      <td>53.936857</td>
      <td>46.735337</td>
      <td>52.270329</td>
      <td>47.162781</td>
      <td>64.626010</td>
      <td>-9.236673</td>
      <td>-10.083801</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 23 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-81c40231-b078-4287-a302-471880fafc8c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-81c40231-b078-4287-a302-471880fafc8c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-81c40231-b078-4287-a302-471880fafc8c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-02ca3da4-7586-4768-a829-e0732c197007">
      <button class="colab-df-quickchart" onclick="quickchart('df-02ca3da4-7586-4768-a829-e0732c197007')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-02ca3da4-7586-4768-a829-e0732c197007 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_b925139d-46e3-496c-b121-e489b9a529b6">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_filtered')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_b925139d-46e3-496c-b121-e489b9a529b6 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_filtered');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
#define subplots
fig,ax = plt.subplots(figsize=(20, 6))

datay1 = df_crns['SWC_CRNS']
datax1 = df_crns['date']

ndvi1 = df_l8r['NDVI']
date1 = df_l8r['date']
ndvi2 = df_l9r['NDVI']
date2 = df_l9r['date']
ndvi3 = df_ndvir['NDVI']
date3 = df_ndvir['date']
ndvi4 = df_merged_ndvi_filtered['NDVI']
date4 = df_merged_ndvi_filtered['date']

ax2 = ax.twinx()

p1 = ax.plot(datax1, datay1, label="CRNS measurements")
p2 = ax2.plot(date1, ndvi1, 's--y',c= 'grey', label="NDVI (Landsat 8)")
p3 = ax2.plot(date2, ndvi2, '>--y',c='grey', label="NDVI (Landsat 9)")
p4 = ax2.plot(date3, ndvi3, 'o--y', c='grey', label="NDVI (MODIS)")
p5 = ax2.plot(date4, ndvi4, 'o-y', c='black', label="Combined NDVI")

lns = p1+p2+p3+p4+p5
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="upper left")

ax.set_xlabel('Date')
ax.set_ylabel('Soil moisture percentage')
ax2.set_ylabel('NDVI')

plt.show()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    
![png](output_29_1.png)
    



```python
# Match specific datasets for comparison. Matching too many at once will lead to loss of information
df_match1 = pd.merge_asof(df_filtered, df_crns, on="date", direction="nearest")
df_match2 = pd.merge_asof(df_match1, df_merged_ndvi_filtered, on='date', direction='nearest')
df_match3 = pd.merge_asof(df_match2, df_lai.sort_values(by='date'), on='date', direction='nearest')

df_matched = df_match3
# match soil moisture predictions to CRNS data
df_matched_crns = df_matched[['date', 'adjusted_VV', 'SSM', 'SSM2', 'SSM22','SSM23','SSM24','SSM210','SSM211','SSM215', 'SSM2_wd','SSM2_wd2', 'SWC_CRNS', 'NDVI', 'LAI']].dropna()
# match soil moisture predictions to point sensor data
df_matched_sensors = df_matched[['date', 'adjusted_VV', 'SSM', 'SSM3','SSM32','SSM33','SSM34','SSM310','SSM311','SSM315', 'SSM3_wd','SSM3_wd2', 'NDVI', 'LAI', 'AVG_of_three_sensors_10cm']].dropna()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
df_matched_ndvi = pd.merge_asof(df_merged_ndvi_filtered, df_match1, on='date', direction='nearest')

df_matched_ndvi_sensors = df_matched_ndvi[['date', 'adjusted_VV', 'SSM', 'SSM2', 'SSM3', 'SWC_CRNS', 'NDVI', 'AVG_of_three_sensors_10cm']].dropna()
df_matched_ndvi_crns = df_matched_ndvi[['date', 'adjusted_VV', 'SSM', 'SSM2', 'SSM3', 'SWC_CRNS', 'NDVI']].dropna()


df_matched_lai = pd.merge_asof(df_lai.sort_values(by='date'), df_match1, on='date', direction='nearest')

df_matched_lai_sensors = df_matched_lai[['date', 'adjusted_VV', 'SSM', 'SSM2', 'SSM3', 'SWC_CRNS', 'LAI', 'AVG_of_three_sensors_10cm']].dropna().drop_duplicates()
df_matched_lai_crns = df_matched_lai[['date', 'adjusted_VV', 'SSM', 'SSM2', 'SSM3', 'SWC_CRNS', 'LAI']].dropna().drop_duplicates()

df_matched_crns_sensor = df_crns.dropna()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
#define subplots
fig,ax = plt.subplots(figsize=(20, 6))

datay1 = df_crns['SWC_CRNS']
datax1 = df_crns['date']
datay2 = df_crns['AVG_of_three_sensors_10cm']
datay3 = df_filtered['SSM2']
datax3 = df_filtered['date']
datay4 = df_matched_crns['SSM24']
datax4 = df_matched_crns['date']
ax2 = ax.twinx()

ax.plot(datax1, datay1)
ax.plot(datax4, datay4, 'o-y', c='black')


plt.show()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    
![png](output_32_1.png)
    



```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

X = df_matched_crns['SSM24']
y = df_matched_crns['SWC_CRNS']
X = df_matched_sensors['SSM34']
y = df_matched_sensors['AVG_of_three_sensors_10cm']

df1 = df_matched_lai_sensors['LAI']
df2 = df_matched_lai_sensors['AVG_of_three_sensors_10cm']
df1 = df_matched_sensors['SSM310']
df2 = df_matched_sensors['AVG_of_three_sensors_10cm']
df1 = df_matched_ndvi_sensors['NDVI']
df2 = df_matched_ndvi_sensors['AVG_of_three_sensors_10cm']

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(df1.to_numpy().reshape(-1, 1)).ravel()
y = scaler.fit_transform(df2.to_numpy().reshape(-1, 1)).ravel()


degree = 1
model = np.poly1d(np.polyfit(X, y, degree))
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = ssreg / sstot

    return results
print(polyfit(X, y, degree))
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_val = rmse(np.array(X), np.array(y))
print("rms error is: " + str(rmse_val))

print(model)

#add fitted polynomial line to scatterplot
fig,ax = plt.subplots(figsize=(6, 6))
polyline = np.linspace(X.min(), X.max())
plt.scatter(X, y, s=2)
plt.plot(polyline, model(polyline))

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(X, y)
ax.annotate(f'R-squared = {r_value**2:.8f}',(30,61))
ax.annotate(f'RMSE = {rmse_val:.8f}',(30,59))
plt.xlabel('X')
plt.ylabel('X')
plt.show()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    {'r_squared': np.float64(0.3356648037604018)}
    rms error is: 0.5524136111818336
     
    -0.4609 x + 0.8327
    


    
![png](output_33_2.png)
    



```python
# Calculate statistics for evaluation of different SSM prediction methods

from scipy.spatial.distance import correlation
from scipy.stats import spearmanr

def statistics(true_value, predicted_values):
  records = []
  for i in predicted_values:
    X = predicted_values[i]
    y = true_value
    nameX = str(i)
    namey = str(df)

    rho, p = spearmanr(X, y)
    corelation = correlation(X, y)

    x_mean = (sum(X))/float(len(X))
    y_mean = (sum(y))/float(len(y))

    top = sum((X-x_mean)*(y-y_mean))
    bottom1 = sum((X-x_mean)**2)
    bottom2 = sum((y-y_mean)**2)
    bottom = (bottom1*bottom2)**(1/2)
    r = top/bottom
    Pr2 = r**2
    r2   = r2_score(y.values, X.values)
    rmse = root_mean_squared_error(y.values, X.values)

    records.append((nameX, namey, Pr2, r2, rmse, rho, corelation))

  cols = ["X", "y", 'Pr2', "r2", "rmse", "rho", "corelation"]
  return (
      pd.DataFrame.from_records(records, columns=cols)
  )

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# print statistics for comapring SSM retrieval with in-field CRNS data
statistics(df_matched_crns['SWC_CRNS'], df_matched_crns[['SSM','SSM2', 'SSM22','SSM23', 'SSM24', 'SSM210', 'SSM211', 'SSM215', 'SSM2_wd', 'SSM2_wd2']])
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-54941d4f-7bc3-4668-9b22-3da581a7dcde" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>y</th>
      <th>Pr2</th>
      <th>r2</th>
      <th>rmse</th>
      <th>rho</th>
      <th>corelation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SSM</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.159378</td>
      <td>-14.811168</td>
      <td>28.172544</td>
      <td>0.412613</td>
      <td>0.600778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SSM2</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.156915</td>
      <td>-0.437367</td>
      <td>8.494306</td>
      <td>0.412613</td>
      <td>0.603875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SSM22</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.156915</td>
      <td>-0.938628</td>
      <td>9.864869</td>
      <td>0.412613</td>
      <td>0.603875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SSM23</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.159378</td>
      <td>-0.579809</td>
      <td>8.905255</td>
      <td>0.412613</td>
      <td>0.600778</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SSM24</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.159378</td>
      <td>-0.185030</td>
      <td>7.712742</td>
      <td>0.412613</td>
      <td>0.600778</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SSM210</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.500198</td>
      <td>0.325439</td>
      <td>5.819084</td>
      <td>0.727283</td>
      <td>0.292753</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SSM211</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.500198</td>
      <td>-3.667498</td>
      <td>15.306862</td>
      <td>0.727283</td>
      <td>0.292753</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SSM215</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.355520</td>
      <td>-0.858253</td>
      <td>9.658206</td>
      <td>0.609424</td>
      <td>0.403745</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SSM2_wd</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.617121</td>
      <td>0.172136</td>
      <td>6.446493</td>
      <td>0.801145</td>
      <td>0.214429</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SSM2_wd2</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.581648</td>
      <td>-0.080558</td>
      <td>7.364922</td>
      <td>0.773346</td>
      <td>0.237342</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-54941d4f-7bc3-4668-9b22-3da581a7dcde')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-54941d4f-7bc3-4668-9b22-3da581a7dcde button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-54941d4f-7bc3-4668-9b22-3da581a7dcde');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-67fcf090-589e-43a6-a9fe-65762efcbfb8">
      <button class="colab-df-quickchart" onclick="quickchart('df-67fcf090-589e-43a6-a9fe-65762efcbfb8')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-67fcf090-589e-43a6-a9fe-65762efcbfb8 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# print statistics for comapring SSM retrieval with in-field point sensor data
statistics(df_matched_sensors['AVG_of_three_sensors_10cm'], df_matched_sensors[['SSM','SSM3', 'SSM32','SSM33', 'SSM34', 'SSM310', 'SSM311', 'SSM315', 'SSM3_wd', 'SSM3_wd2']])
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>







  <div id="df-3d4bf84f-7a2a-43f6-b913-e8902a5ddd84" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>y</th>
      <th>Pr2</th>
      <th>r2</th>
      <th>rmse</th>
      <th>rho</th>
      <th>corelation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SSM</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.063910</td>
      <td>-11.809322</td>
      <td>29.050012</td>
      <td>0.207003</td>
      <td>0.747195</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SSM3</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.062397</td>
      <td>-0.286632</td>
      <td>9.206836</td>
      <td>0.207003</td>
      <td>0.750206</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SSM32</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.062397</td>
      <td>-1.706543</td>
      <td>13.353365</td>
      <td>0.207003</td>
      <td>0.750206</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SSM33</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.063910</td>
      <td>-0.295591</td>
      <td>9.238835</td>
      <td>0.207003</td>
      <td>0.747195</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SSM34</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.063910</td>
      <td>-0.782890</td>
      <td>10.837908</td>
      <td>0.207003</td>
      <td>0.747195</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SSM310</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.555744</td>
      <td>0.452383</td>
      <td>6.006505</td>
      <td>0.825770</td>
      <td>0.254518</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SSM311</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.555744</td>
      <td>-0.356696</td>
      <td>9.454195</td>
      <td>0.825770</td>
      <td>0.254518</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SSM315</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.296928</td>
      <td>0.152629</td>
      <td>7.471712</td>
      <td>0.665546</td>
      <td>0.455089</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SSM3_wd</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.843918</td>
      <td>0.841765</td>
      <td>3.228752</td>
      <td>0.896919</td>
      <td>0.081350</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SSM3_wd2</td>
      <td>0     longitude   latitude                    ...</td>
      <td>0.663639</td>
      <td>0.542713</td>
      <td>5.488805</td>
      <td>0.643978</td>
      <td>0.185359</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3d4bf84f-7a2a-43f6-b913-e8902a5ddd84')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3d4bf84f-7a2a-43f6-b913-e8902a5ddd84 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3d4bf84f-7a2a-43f6-b913-e8902a5ddd84');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-42c004ff-8e82-4e17-bb50-6c7d2b421e62">
      <button class="colab-df-quickchart" onclick="quickchart('df-42c004ff-8e82-4e17-bb50-6c7d2b421e62')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-42c004ff-8e82-4e17-bb50-6c7d2b421e62 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
raise SystemExit("Stop right there!")

# After this part some attempts were done on the adjustment of SSM based on vegetation data (through NDVI and LAI)
# This however was done on a rudimentary level do to time restrictions and did not provide any satisfactory results
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    An exception has occurred, use %tb to see the full traceback.
    

    SystemExit: Stop right there!
    



```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------------
# 1️⃣  (i, j, k) generator on a 0.01 grid with i + j + k = 1
# ---------------------------------------------------------------
def triples_sum_to_one(step=0.01):
    scale = round(1 / step)                       # 0.01 → 100
    for i_int in range(scale + 1):
        for j_int in range(scale + 1):
            k_int = scale - i_int - j_int         # keep the sum = 1
            if 0 <= k_int <= scale:
                yield i_int / scale, j_int / scale, k_int / scale


# ---------------------------------------------------------------
# 2️⃣  helper: scale a Series to (–1, 1)
# ---------------------------------------------------------------
def scale_minus1_to_1(series):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(series.to_numpy().reshape(-1, 1)).ravel()


# ---------------------------------------------------------------
# 3️⃣  main evaluation routine  (-1,1 scaling  ➜  new coefficients)
# ---------------------------------------------------------------
def evaluate_triples(df_matched_sensors, df_X, step=0.01):
    # a) individually scale the three raw variables to (-1, 1)
    SSM_scaled  = scale_minus1_to_1(df_matched_sensors["SSM310"])
    NDVI_scaled = scale_minus1_to_1(df_matched_sensors["NDVI"])
    LAI_scaled  = scale_minus1_to_1(df_matched_sensors["LAI"])

    y_true = df_X.to_numpy()
    y_min, y_max = y_true.min(), y_true.max()

    records = []
    for i, j, k in triples_sum_to_one(step):
        y_pred_raw = (
              i * ( SSM_scaled )
            + j * (-0.4609 * NDVI_scaled + 0.8327 )
            + k * (-0.4131 * LAI_scaled  + 0.7563 )
        )

        # 0.7208 x + 0.1885 #SSM310
        # 0.9039 * SSM_scaled  + 0.004006 #SSM3_wd


        # (optional) map the prediction back to the span of df_X
        y_pred_scaled = (y_pred_raw - y_pred_raw.min()) / (y_pred_raw.max() - y_pred_raw.min())
        y_pred = y_pred_scaled * (y_max - y_min) + y_min

        mse  = mean_squared_error(y_true, y_pred)   # works on any scikit-learn version
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)

        records.append((i, j, k, r2, rmse))

    cols = ["i", "j", "k", "r_squared", "rmse"]
    return (
        pd.DataFrame.from_records(records, columns=cols)
          .sort_values("i")
          .reset_index(drop=True)
    )
```


```python
results = evaluate_triples(df_matched_sensors, df_matched_sensors['AVG_of_three_sensors_10cm'])
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(results)
```


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------
# 1️⃣  (i, j, k) generator on a 0.01 grid with i + j + k = 1
# ---------------------------------------------------------------
def triples_sum_to_one(step=0.01):
    scale = round(1 / step)                       # 0.01 → 100
    for i_int in range(scale + 1):
        for j_int in range(scale + 1):
            k_int = scale - i_int - j_int         # keep the sum = 1
            if 0 <= k_int <= scale:
                yield i_int / scale, j_int / scale, k_int / scale


# ---------------------------------------------------------------
# 2️⃣  helper: scale a Series to (–1, 1)
# ---------------------------------------------------------------
def scale_minus1_to_1(series):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(series.to_numpy().reshape(-1, 1)).ravel()


# ---------------------------------------------------------------
# 3️⃣  main evaluation routine  (-1,1 scaling  ➜  new coefficients)
# ---------------------------------------------------------------
def evaluate_triples(df_matched_crns, df_X, step=0.01):
    # a) individually scale the three raw variables to (-1, 1)
    SSM_scaled  = scale_minus1_to_1(df_matched_crns["SSM2_wd"])
    NDVI_scaled = scale_minus1_to_1(df_matched_crns["NDVI"])
    LAI_scaled  = scale_minus1_to_1(df_matched_crns["LAI"])

    y_true = df_X.to_numpy()
    y_min, y_max = y_true.min(), y_true.max()

    records = []
    for i, j, k in triples_sum_to_one(step):
        y_pred_raw = (
              i * (0.688 * SSM_scaled + 0.1684)
              + j * (-0.2587 * NDVI_scaled   + 0.7527)
              + k * (-0.3811  * LAI_scaled    + 0.6832)
        )

        # (optional) map the prediction back to the span of df_X
        y_pred_scaled = (y_pred_raw - y_pred_raw.min()) / (y_pred_raw.max() - y_pred_raw.min())
        y_pred = y_pred_scaled * (y_max - y_min) + y_min

        mse  = mean_squared_error(y_true, y_pred)   # works on any scikit-learn version
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)

        records.append((i, j, k, r2, rmse))

    cols = ["i", "j", "k", "r_squared", "rmse"]
    return (
        pd.DataFrame.from_records(records, columns=cols)
          .sort_values("r_squared")
          .reset_index(drop=True)
    )
```


```python
results = evaluate_triples(df_matched_crns, df_matched_crns['SWC_CRNS'])
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(results)
```


```python
df_matched_sensors['model310'] = \
  [((0.85*(x)) + \
  (0.15*(-0.4609*y + 0.8327)) + \
  (0*(-0.5279*z + 0.8791))) \
  for x,y,z in zip(df_matched_sensors['SSM310'], df_matched_sensors['NDVI'], df_matched_sensors['LAI'])]
```


```python
X = df_matched_sensors['SSM3_wd']
y = df_matched_sensors['AVG_of_three_sensors_10cm']
X = df_matched_sensors['model310']
y = df_matched_sensors['AVG_of_three_sensors_10cm']

degree = 1
model = np.poly1d(np.polyfit(X, y, degree))
print(model)

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = ssreg / sstot

    return results
print(polyfit(X, y, degree))

#add fitted polynomial line to scatterplot
fig,ax = plt.subplots(figsize=(8, 8))
polyline = np.linspace(X.min(), X.max())
plt.scatter(X, y)
plt.plot(polyline, model(polyline))
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_val = rmse(np.array(X), np.array(y))
print("rms error is: " + str(rmse_val))
slope, intercept, r_value, p_value, std_err = linregress(X, y)
plt.annotate(f'R-squared = {r_value**2:.8f}',(30,58))
plt.annotate(f'RMSE = {rmse_val:.8f}',(30,56))
plt.xlabel('Soil moisture value by CRNS')
plt.ylabel('Soil moisture value by point sensors')
plt.show()

```
