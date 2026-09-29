// 1. Load your cleaned points asset
var points = ee.FeatureCollection('projects/durable-pulsar-473408-r2/assets/cleaned_tobacco_barn_coordinates');

// 2. Get the overall bounding region of all 16,000 points (prevents memory overhead)
var areaOfInterest = points.geometry().bounds();

// 3. Filter Sentinel-2 collection across the bounding area
var composite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2024-01-01', '2024-12-31')
  .filterBounds(areaOfInterest)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .median()
  .select(['B4', 'B3', 'B2']); // Red, Green, Blue bands

// 4. Create 100m bounding patches centered on each point
var patches = points.map(function(feature) {
  return feature.geometry().buffer(50).bounds();
});

// 5. Add layers to the map
Map.centerObject(points, 7);

// Render imagery with visualization parameters
Map.addLayer(composite, {min: 0, max: 3000, bands: ['B4', 'B3', 'B2']}, 'Sentinel-2 Satellite Image');
Map.addLayer(points, {color: 'yellow'}, '6230 Tobacco Barn Points');

// Export task with increased pixel limit and automatic tiling
Export.image.toDrive({
  image: composite.visualize({min: 0, max: 3000}),
  description: 'TRB_Tobacco_Barn_Image_Patches',
  folder: 'TRB_Barn_Patches',
  scale: 10,
  region: areaOfInterest,
  maxPixels: 1e13, // Allows large area exports up to 10 trillion pixels
  shardSize: 256,  // Automatically splits output into manageable tiled image files
  fileFormat: 'GeoTIFF'
});
