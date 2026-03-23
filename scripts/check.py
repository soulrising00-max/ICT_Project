# run this one-off to find the right tile
import geopandas as gpd

gdf = gpd.read_file("data/polygons/562_KasigauCorridorREDD.geojson")
print(gdf.total_bounds)  # minx, miny, maxx, maxy
