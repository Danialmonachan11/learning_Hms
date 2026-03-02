import fastlibrary as fl
import zygorw as zr
import pandas as pd

from color_map import false_color_map_with_histogram

zygo = zr.ZygoFile("T:/asm/EXE_ZdMB/10_EXE_MB_Flatness_Database/EX1-B3028/2024-10-03 - 09h29 - after HI bonding #4/01 - Side data/EX1-B3028_$Seite_ZA$B$_M1_nHI.wrk")
#print(zygo)

#zygo.readFile() 
#topo = fl.Topography(zygo)
#topo = fl.level(topo)
#label_map, labels = fl.label(topo)
#label_map.update_zygo_file(zygo)


test_file=R"T:\asm\EXE_ZdMB\10_EXE_MB_Flatness_Database\EX1-B3028\2024-10-03 - 09h29 - after HI bonding #4\01 - Side data\EX1-B3028_$Seite_ZA$B$_M1_nHI.wrk"
test_data = fl.read_zygo(test_file)
print(test_data)
wrk_image=false_color_map_with_histogram(test_data, title="Test data", z_min=-50, z_max=50)

print(wrk_image)