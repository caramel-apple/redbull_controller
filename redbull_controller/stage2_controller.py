"""Stage 2 Competition Controller: obstacle-aware pure pursuit + LiDAR path guard + follow-the-gap."""
import csv
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


# Rear-axle race line for the official Stage 2 map (x, y, footprint clearance m, body-yaw offset rad),
# generated offline from maps/levine_obs.png + maps/levine_centerline.csv (counter-clockwise).
RACE_LINE = """
0.0372,-0.4169,0.1795,-0.0562
0.1577,-0.3661,0.1125,-0.0759
0.2768,-0.2828,0.1114,-0.0764
0.3965,-0.1977,0.1123,-0.0679
0.5204,-0.1444,0.1090,-0.0407
0.6445,-0.0973,0.0933,0.0049
0.7722,-0.0648,0.0654,0.0335
0.9000,-0.0561,0.0575,0.0453
1.0259,-0.0561,0.0599,0.0528
1.1509,-0.0578,0.0631,0.0415
1.2759,-0.0588,0.0658,0.0166
1.4010,-0.0564,0.0693,-0.0061
1.5260,-0.0495,0.0721,-0.0250
1.6512,-0.0478,0.0648,-0.0219
1.7765,-0.0522,0.0627,-0.0038
1.9011,-0.0647,0.0600,0.0164
2.0250,-0.1127,0.1132,0.0390
2.1488,-0.1511,0.1842,0.0540
2.2727,-0.2046,0.2324,0.0475
2.3983,-0.2440,0.2599,0.0282
2.5255,-0.2727,0.2880,0.0077
2.6534,-0.2878,0.3374,-0.0161
2.7812,-0.2885,0.4070,-0.0329
2.9080,-0.2788,0.4872,-0.0460
3.0338,-0.2615,0.5759,-0.0526
3.1590,-0.2378,0.5942,-0.0536
3.2837,-0.2120,0.6143,-0.0504
3.4090,-0.1858,0.6209,-0.0429
3.5344,-0.1612,0.6105,-0.0312
3.6603,-0.1404,0.5959,-0.0179
3.7865,-0.1220,0.5873,-0.0047
3.9130,-0.1073,0.5756,0.0074
4.0393,-0.0941,0.5637,0.0153
4.1658,-0.0824,0.5556,0.0200
4.2921,-0.0720,0.5277,0.0227
4.4184,-0.0586,0.5251,0.0223
4.5450,-0.0474,0.5022,0.0198
4.6711,-0.0339,0.4844,0.0180
4.7974,-0.0194,0.4657,0.0163
4.9236,-0.0046,0.4441,0.0144
5.0498,0.0118,0.4283,0.0144
5.1760,0.0278,0.4056,0.0140
5.3021,0.0450,0.3839,0.0131
5.4282,0.0628,0.3584,0.0124
5.5541,0.0824,0.3336,0.0095
5.6800,0.1033,0.3042,0.0047
5.8056,0.1242,0.2597,-0.0008
5.9311,0.1486,0.2398,-0.0053
6.0568,0.1723,0.2209,-0.0096
6.1822,0.1971,0.2083,-0.0126
6.3076,0.2190,0.1985,-0.0126
6.4331,0.2402,0.1920,-0.0098
6.5589,0.2591,0.1833,-0.0050
6.6846,0.2744,0.1787,-0.0011
6.8104,0.2869,0.1757,0.0024
6.9360,0.2966,0.1724,0.0044
7.0617,0.3069,0.1681,0.0028
7.1873,0.3154,0.1665,-0.0020
7.3128,0.3242,0.1637,-0.0071
7.4386,0.3326,0.1627,-0.0104
7.5641,0.3390,0.1623,-0.0131
7.6900,0.3421,0.1623,-0.0148
7.8167,0.3381,0.1657,-0.0142
7.9434,0.3296,0.1435,-0.0113
8.0699,0.3128,0.1247,-0.0050
8.1959,0.2868,0.0997,0.0049
8.3211,0.2517,0.0726,0.0168
8.4452,0.2012,0.0657,0.0368
8.5696,0.1448,0.0594,0.0553
8.6952,0.0832,0.0493,0.0581
8.8290,0.0054,0.0668,0.0512
8.9661,-0.0465,0.1531,0.0386
9.1050,-0.0776,0.2786,0.0136
9.2427,-0.0842,0.2458,-0.0119
9.3796,-0.0699,0.1764,-0.0250
9.5119,-0.0328,0.1348,-0.0311
9.6402,0.0208,0.0957,-0.0309
9.7601,0.0943,0.0778,-0.0260
9.8715,0.1849,0.0769,-0.0188
9.9696,0.2918,0.0779,-0.0117
10.0502,0.4103,0.0679,-0.0042
10.1121,0.5359,0.0756,0.0004
10.1538,0.6665,0.1069,0.0015
10.1768,0.7985,0.1601,-0.0047
10.1785,0.9321,0.2169,-0.0143
10.1655,1.0639,0.2334,-0.0218
10.1292,1.1975,0.2737,-0.0247
10.0887,1.3307,0.3127,-0.0210
10.0476,1.4645,0.3510,-0.0066
10.0152,1.5981,0.3801,0.0148
9.9921,1.7325,0.3790,0.0348
9.9820,1.8671,0.3736,0.0529
9.9805,2.0011,0.3742,0.0664
9.9853,2.1349,0.3780,0.0727
9.9942,2.2685,0.3772,0.0713
10.0040,2.4018,0.3770,0.0649
10.0123,2.5352,0.3774,0.0535
10.0184,2.6687,0.3781,0.0386
10.0223,2.8023,0.3791,0.0241
10.0226,2.9360,0.3819,0.0112
10.0208,3.0695,0.3851,-0.0004
10.0180,3.2030,0.3881,-0.0087
10.0135,3.3365,0.3921,-0.0134
10.0083,3.4698,0.3962,-0.0166
10.0029,3.6029,0.3999,-0.0186
9.9974,3.7361,0.4035,-0.0191
9.9913,3.8693,0.4076,-0.0187
9.9847,4.0023,0.4120,-0.0182
9.9781,4.1352,0.4162,-0.0178
9.9716,4.2680,0.4202,-0.0167
9.9647,4.4007,0.4247,-0.0150
9.9573,4.5334,0.4297,-0.0138
9.9501,4.6659,0.4344,-0.0127
9.9431,4.7982,0.4390,-0.0109
9.9353,4.9304,0.4444,-0.0094
9.9280,5.0624,0.4495,-0.0085
9.9204,5.1943,0.4553,-0.0069
9.9127,5.3260,0.4611,-0.0048
9.9052,5.4576,0.4668,-0.0033
9.8982,5.5889,0.4722,-0.0014
9.8913,5.7201,0.4780,0.0010
9.8841,5.8511,0.4844,0.0029
9.8774,5.9818,0.4904,0.0044
9.8709,6.1124,0.4964,0.0062
9.8642,6.2427,0.5031,0.0091
9.8574,6.3728,0.5101,0.0118
9.8505,6.5024,0.5173,0.0153
9.8443,6.6313,0.5243,0.0204
9.8362,6.7589,0.5338,0.0274
9.8284,6.8849,0.5435,0.0354
9.8185,7.0087,0.5559,0.0453
9.8080,7.1303,0.5700,0.0571
9.7938,7.2488,0.5462,0.0673
9.7764,7.3641,0.5228,0.0774
9.7541,7.4754,0.4872,0.0839
9.7244,7.5817,0.4749,0.0883
9.6899,7.6828,0.4337,0.0851
9.6450,7.7761,0.4344,0.0801
9.5983,7.8638,0.3923,0.0683
9.5396,7.9403,0.3978,0.0544
9.4825,8.0120,0.3596,0.0388
9.4135,8.0705,0.3684,0.0249
9.3473,8.1281,0.3473,0.0108
9.2703,8.1723,0.3525,-0.0017
9.1926,8.2164,0.3503,-0.0103
9.1070,8.2487,0.3535,-0.0218
9.0183,8.2785,0.3699,-0.0321
8.9248,8.2992,0.3925,-0.0417
8.8272,8.3147,0.4152,-0.0472
8.7263,8.3236,0.4314,-0.0537
8.6215,8.3269,0.4435,-0.0543
8.5143,8.3265,0.4520,-0.0517
8.4036,8.3212,0.4547,-0.0455
8.2909,8.3155,0.4284,-0.0375
8.1754,8.3060,0.4284,-0.0268
8.0579,8.2984,0.4045,-0.0134
7.9388,8.2885,0.4031,0.0010
7.8169,8.2809,0.3932,0.0151
7.6954,8.2748,0.3840,0.0293
7.5728,8.2698,0.3766,0.0443
7.4493,8.2668,0.3709,0.0567
7.3249,8.2661,0.3694,0.0682
7.2006,8.2676,0.3598,0.0782
7.0763,8.2698,0.3547,0.0863
6.9520,8.2734,0.3382,0.0925
6.8272,8.2758,0.3328,0.0946
6.7023,8.2788,0.3131,0.0931
6.5777,8.2793,0.2920,0.0873
6.4524,8.2758,0.2775,0.0736
6.3273,8.2699,0.2240,0.0548
6.2024,8.2581,0.1925,0.0363
6.0759,8.2412,0.1811,0.0182
5.9491,8.2248,0.1747,0.0041
5.8219,8.2093,0.1679,-0.0010
5.6940,8.1978,0.1637,0.0010
5.5657,8.1923,0.1628,0.0067
5.4373,8.1941,0.1682,0.0161
5.3089,8.2032,0.2073,0.0273
5.1807,8.2173,0.2396,0.0364
5.0526,8.2378,0.2644,0.0424
4.9241,8.2618,0.2841,0.0456
4.7956,8.2868,0.3035,0.0449
4.6664,8.3127,0.3241,0.0396
4.5367,8.3361,0.3428,0.0320
4.4066,8.3573,0.3609,0.0213
4.2761,8.3741,0.3758,0.0091
4.1457,8.3878,0.3892,-0.0029
4.0154,8.3959,0.3988,-0.0147
3.8854,8.4024,0.4080,-0.0249
3.7558,8.4048,0.4140,-0.0321
3.6266,8.4056,0.4191,-0.0355
3.4977,8.4052,0.4233,-0.0365
3.3693,8.4041,0.4270,-0.0349
3.2412,8.4041,0.4313,-0.0309
3.1132,8.4033,0.4343,-0.0254
2.9854,8.4035,0.4380,-0.0205
2.8578,8.4047,0.4418,-0.0156
2.7303,8.4062,0.4455,-0.0097
2.6032,8.4078,0.4487,-0.0048
2.4761,8.4103,0.4526,-0.0014
2.3494,8.4133,0.4564,0.0019
2.2228,8.4165,0.4599,0.0053
2.0963,8.4189,0.4625,0.0068
1.9699,8.4221,0.4660,0.0071
1.8439,8.4255,0.4691,0.0082
1.7179,8.4275,0.4709,0.0089
1.5920,8.4300,0.4732,0.0080
1.4664,8.4325,0.4752,0.0078
1.3408,8.4346,0.4769,0.0086
1.2156,8.4363,0.4778,0.0089
1.0901,8.4382,0.4792,0.0087
0.9648,8.4405,0.4807,0.0089
0.8397,8.4423,0.4816,0.0091
0.7146,8.4437,0.4820,0.0085
0.5898,8.4451,0.4825,0.0077
0.4649,8.4462,0.4829,0.0072
0.3401,8.4472,0.4832,0.0063
0.2153,8.4480,0.4827,0.0056
0.0907,8.4486,0.4797,0.0054
-0.0339,8.4487,0.4809,0.0053
-0.1585,8.4493,0.4820,0.0048
-0.2830,8.4498,0.4818,0.0048
-0.4075,8.4502,0.4773,0.0045
-0.5319,8.4498,0.4787,0.0033
-0.6564,8.4503,0.4737,0.0028
-0.7808,8.4494,0.4757,0.0025
-0.9053,8.4494,0.4770,0.0019
-1.0296,8.4493,0.4749,0.0014
-1.1541,8.4489,0.4737,0.0014
-1.2785,8.4486,0.4735,0.0007
-1.4029,8.4483,0.4739,0.0006
-1.5274,8.4484,0.4743,0.0014
-1.6519,8.4485,0.4742,0.0026
-1.7764,8.4492,0.4742,0.0043
-1.9009,8.4501,0.4741,0.0072
-2.0255,8.4511,0.4708,0.0107
-2.1502,8.4520,0.4712,0.0141
-2.2749,8.4532,0.4657,0.0171
-2.3997,8.4533,0.4601,0.0194
-2.5245,8.4525,0.4448,0.0199
-2.6495,8.4494,0.4200,0.0198
-2.7745,8.4448,0.3983,0.0182
-2.8996,8.4371,0.3700,0.0154
-3.0248,8.4285,0.3588,0.0123
-3.1501,8.4154,0.3393,0.0092
-3.2754,8.4015,0.3241,0.0049
-3.4010,8.3845,0.2969,0.0009
-3.5267,8.3671,0.2878,-0.0004
-3.6526,8.3468,0.2715,-0.0009
-3.7787,8.3273,0.2519,-0.0008
-3.9051,8.3078,0.2311,-0.0003
-4.0317,8.2879,0.2120,-0.0009
-4.1583,8.2680,0.1931,-0.0048
-4.2850,8.2471,0.1745,-0.0119
-4.4120,8.2258,0.1624,-0.0205
-4.5390,8.2055,0.1559,-0.0266
-4.6664,8.1861,0.1379,-0.0299
-4.7942,8.1737,0.1232,-0.0275
-4.9221,8.1702,0.1169,-0.0181
-5.0500,8.1748,0.1153,-0.0051
-5.1770,8.1937,0.1416,0.0097
-5.3033,8.2196,0.2248,0.0237
-5.4285,8.2562,0.2950,0.0331
-5.5532,8.2990,0.3344,0.0373
-5.6780,8.3440,0.3759,0.0367
-5.8030,8.3904,0.4184,0.0293
-5.9287,8.4345,0.4587,0.0189
-6.0555,8.4752,0.4966,0.0071
-6.1829,8.5112,0.5310,-0.0049
-6.3106,8.5436,0.5638,-0.0150
-6.4386,8.5725,0.5942,-0.0220
-6.5666,8.5999,0.5906,-0.0258
-6.6946,8.6253,0.5592,-0.0263
-6.8227,8.6513,0.5377,-0.0241
-6.9507,8.6763,0.5037,-0.0200
-7.0786,8.7032,0.4837,-0.0146
-7.2067,8.7291,0.4539,-0.0090
-7.3345,8.7559,0.4303,-0.0040
-7.4626,8.7820,0.4075,-0.0015
-7.5906,8.8074,0.3870,-0.0001
-7.7185,8.8317,0.3735,0.0005
-7.8466,8.8533,0.3486,-0.0008
-7.9748,8.8753,0.3307,-0.0035
-8.1029,8.8958,0.3128,-0.0040
-8.2312,8.9149,0.2929,-0.0051
-8.3594,8.9344,0.2719,-0.0042
-8.4877,8.9549,0.2647,0.0028
-8.6160,8.9738,0.2178,0.0108
-8.7444,8.9981,0.1944,0.0176
-8.8729,9.0220,0.1874,0.0243
-9.0011,9.0440,0.1808,0.0267
-9.1291,9.0634,0.1768,0.0220
-9.2578,9.0789,0.1750,0.0158
-9.3865,9.0905,0.1757,0.0091
-9.5147,9.0976,0.1833,0.0020
-9.6433,9.1034,0.1889,-0.0038
-9.7720,9.1057,0.1932,-0.0069
-9.9006,9.1069,0.1949,-0.0088
-10.0289,9.1073,0.1948,-0.0087
-10.1571,9.1075,0.1954,-0.0048
-10.2857,9.1079,0.1966,0.0022
-10.4144,9.1098,0.1978,0.0130
-10.5430,9.1126,0.1974,0.0259
-10.6713,9.1163,0.1960,0.0375
-10.7996,9.1199,0.1801,0.0446
-10.9277,9.1191,0.1735,0.0464
-11.0554,9.1132,0.1647,0.0399
-11.1823,9.0984,0.1648,0.0274
-11.3082,9.0761,0.1708,0.0116
-11.4327,9.0435,0.2179,-0.0048
-11.5562,9.0055,0.3076,-0.0183
-11.6789,8.9607,0.3672,-0.0263
-11.8010,8.9144,0.4099,-0.0275
-11.9238,8.8681,0.4531,-0.0214
-12.0482,8.8232,0.4955,-0.0089
-12.1739,8.7814,0.5363,0.0069
-12.3001,8.7429,0.5756,0.0240
-12.4262,8.7060,0.5764,0.0401
-12.5514,8.6699,0.5629,0.0535
-12.6752,8.6316,0.5522,0.0633
-12.7967,8.5901,0.5390,0.0699
-12.9156,8.5427,0.5320,0.0736
-13.0305,8.4890,0.5268,0.0729
-13.1406,8.4270,0.5278,0.0696
-13.2449,8.3570,0.5318,0.0643
-13.3417,8.2777,0.5404,0.0552
-13.4299,8.1900,0.4866,0.0449
-13.5082,8.0943,0.4439,0.0348
-13.5765,7.9918,0.4188,0.0233
-13.6329,7.8838,0.4030,0.0122
-13.6774,7.7726,0.3953,0.0019
-13.7115,7.6592,0.3909,-0.0088
-13.7369,7.5442,0.3920,-0.0175
-13.7538,7.4277,0.3998,-0.0229
-13.7640,7.3088,0.4088,-0.0258
-13.7691,7.1879,0.4189,-0.0258
-13.7697,7.0653,0.4299,-0.0228
-13.7672,6.9410,0.4365,-0.0174
-13.7614,6.8145,0.4367,-0.0110
-13.7544,6.6855,0.4394,-0.0036
-13.7473,6.5552,0.4436,0.0057
-13.7392,6.4231,0.4504,0.0168
-13.7334,6.2901,0.4568,0.0284
-13.7276,6.1559,0.4650,0.0404
-13.7254,6.0210,0.4712,0.0518
-13.7243,5.8864,0.4783,0.0616
-13.7260,5.7520,0.4840,0.0695
-13.7285,5.6176,0.4906,0.0748
-13.7330,5.4827,0.4957,0.0782
-13.7385,5.3481,0.5006,0.0792
-13.7453,5.2137,0.5048,0.0783
-13.7527,5.0798,0.5147,0.0760
-13.7600,4.9457,0.5537,0.0721
-13.7675,4.8117,0.6185,0.0661
-13.7751,4.6780,0.7024,0.0599
-13.7819,4.5446,0.6346,0.0534
-13.7877,4.4112,0.5701,0.0452
-13.7937,4.2780,0.5376,0.0375
-13.7979,4.1450,0.5316,0.0308
-13.8011,4.0120,0.5331,0.0236
-13.8041,3.8793,0.5356,0.0167
-13.8058,3.7467,0.5384,0.0113
-13.8064,3.6143,0.5408,0.0060
-13.8060,3.4820,0.5433,0.0007
-13.8051,3.3499,0.5456,-0.0034
-13.8030,3.2181,0.5485,-0.0067
-13.8003,3.0863,0.5515,-0.0093
-13.7968,2.9548,0.5548,-0.0110
-13.7927,2.8235,0.5578,-0.0121
-13.7886,2.6923,0.5605,-0.0124
-13.7837,2.5613,0.5638,-0.0114
-13.7783,2.4306,0.5674,-0.0099
-13.7729,2.3000,0.5708,-0.0080
-13.7673,2.1697,0.5741,-0.0045
-13.7613,2.0395,0.5781,0.0007
-13.7547,1.9097,0.5828,0.0063
-13.7479,1.7800,0.5878,0.0131
-13.7408,1.6511,0.5936,0.0217
-13.7318,1.5231,0.5830,0.0306
-13.7221,1.3966,0.5515,0.0400
-13.7092,1.2721,0.5209,0.0492
-13.6937,1.1500,0.4969,0.0569
-13.6745,1.0306,0.4778,0.0639
-13.6504,0.9146,0.4599,0.0702
-13.6208,0.8026,0.4475,0.0729
-13.5860,0.6950,0.4361,0.0725
-13.5446,0.5926,0.4258,0.0709
-13.4957,0.4967,0.4234,0.0653
-13.4413,0.4068,0.4238,0.0568
-13.3805,0.3236,0.4269,0.0483
-13.3137,0.2475,0.4336,0.0379
-13.2412,0.1778,0.4483,0.0261
-13.1634,0.1148,0.4732,0.0159
-13.0800,0.0586,0.5072,0.0039
-12.9909,0.0090,0.5025,-0.0093
-12.8978,-0.0351,0.4908,-0.0177
-12.7995,-0.0710,0.4742,-0.0251
-12.6966,-0.1021,0.4565,-0.0319
-12.5914,-0.1288,0.4544,-0.0339
-12.4821,-0.1495,0.4382,-0.0324
-12.3703,-0.1683,0.4372,-0.0302
-12.2557,-0.1829,0.4264,-0.0246
-12.1392,-0.1965,0.4258,-0.0170
-12.0207,-0.2077,0.4170,-0.0087
-11.9008,-0.2189,0.4162,0.0012
-11.7796,-0.2291,0.4083,0.0121
-11.6571,-0.2396,0.4019,0.0238
-11.5341,-0.2506,0.3947,0.0352
-11.4105,-0.2627,0.3887,0.0470
-11.2870,-0.2757,0.3877,0.0585
-11.1625,-0.2887,0.3785,0.0683
-11.0371,-0.3035,0.3759,0.0760
-10.9120,-0.3184,0.3741,0.0827
-10.7863,-0.3331,0.3728,0.0869
-10.6598,-0.3475,0.3716,0.0872
-10.5331,-0.3612,0.3735,0.0847
-10.4059,-0.3729,0.3773,0.0797
-10.2783,-0.3828,0.3792,0.0725
-10.1500,-0.3901,0.3810,0.0638
-10.0211,-0.3963,0.3827,0.0542
-9.8918,-0.4007,0.3853,0.0446
-9.7621,-0.4046,0.3873,0.0352
-9.6323,-0.4078,0.3776,0.0252
-9.5018,-0.4131,0.3568,0.0195
-9.3715,-0.4203,0.3388,0.0153
-9.2406,-0.4311,0.2835,0.0163
-9.1100,-0.4528,0.2351,0.0257
-8.9796,-0.4732,0.1434,0.0341
-8.8490,-0.5119,0.1091,0.0380
-8.7187,-0.5513,0.1052,0.0399
-8.5878,-0.5894,0.0955,0.0356
-8.4565,-0.6237,0.0879,0.0216
-8.3247,-0.6510,0.0839,0.0079
-8.1930,-0.6717,0.0879,-0.0041
-8.0614,-0.6844,0.0994,-0.0130
-7.9308,-0.6912,0.1066,-0.0185
-7.8006,-0.6952,0.1115,-0.0209
-7.6710,-0.6966,0.1163,-0.0197
-7.5420,-0.6972,0.1198,-0.0168
-7.4133,-0.6963,0.1208,-0.0130
-7.2848,-0.6962,0.1223,-0.0078
-7.1567,-0.6963,0.1220,-0.0005
-7.0289,-0.6978,0.1230,0.0087
-6.9012,-0.6997,0.1231,0.0172
-6.7740,-0.7020,0.1166,0.0210
-6.6470,-0.7029,0.1048,0.0215
-6.5200,-0.6983,0.0861,0.0155
-6.3938,-0.6838,0.0846,0.0035
-6.2683,-0.6653,0.0824,-0.0086
-6.1438,-0.6323,0.1209,-0.0188
-6.0199,-0.5989,0.2213,-0.0258
-5.8965,-0.5590,0.2774,-0.0268
-5.7731,-0.5208,0.3115,-0.0225
-5.6492,-0.4847,0.3440,-0.0160
-5.5249,-0.4542,0.3709,-0.0060
-5.4002,-0.4287,0.3940,0.0034
-5.2750,-0.4106,0.4110,0.0117
-5.1500,-0.3974,0.4250,0.0195
-5.0252,-0.3898,0.4344,0.0253
-4.9005,-0.3851,0.4419,0.0279
-4.7761,-0.3845,0.4458,0.0296
-4.6518,-0.3841,0.4500,0.0291
-4.5275,-0.3846,0.4529,0.0261
-4.4035,-0.3855,0.4551,0.0214
-4.2795,-0.3849,0.4580,0.0160
-4.1557,-0.3841,0.4604,0.0091
-4.0318,-0.3818,0.4638,0.0024
-3.9081,-0.3784,0.4677,-0.0026
-3.7847,-0.3741,0.4718,-0.0073
-3.6613,-0.3692,0.4758,-0.0107
-3.5379,-0.3650,0.4792,-0.0116
-3.4146,-0.3590,0.4843,-0.0116
-3.2916,-0.3541,0.4879,-0.0122
-3.1685,-0.3493,0.4915,-0.0119
-3.0455,-0.3439,0.4956,-0.0114
-2.9225,-0.3383,0.4999,-0.0129
-2.7994,-0.3327,0.5041,-0.0144
-2.6766,-0.3268,0.5084,-0.0162
-2.5537,-0.3203,0.5129,-0.0194
-2.4308,-0.3149,0.5163,-0.0229
-2.3080,-0.3091,0.5206,-0.0251
-2.1850,-0.3041,0.5238,-0.0267
-2.0623,-0.3008,0.5136,-0.0270
-1.9393,-0.2998,0.5019,-0.0251
-1.8163,-0.3015,0.4876,-0.0217
-1.6933,-0.3058,0.4717,-0.0181
-1.5702,-0.3129,0.4515,-0.0139
-1.4470,-0.3230,0.4011,-0.0109
-1.3238,-0.3349,0.3399,-0.0083
-1.2006,-0.3495,0.3159,-0.0051
-1.0773,-0.3643,0.3181,-0.0039
-0.9541,-0.3819,0.3227,0.0044
-0.8307,-0.4004,0.3074,0.0133
-0.7079,-0.4169,0.2965,0.0297
-0.5844,-0.4436,0.3091,0.0465
-0.4618,-0.4517,0.2670,0.0526
-0.3372,-0.4803,0.2655,0.0437
-0.2113,-0.4732,0.2757,0.0246
-0.0864,-0.4628,0.1787,-0.0101
"""

LIDAR_X = 0.275             # LiDAR ahead of rear axle (sim.yaml lidar_base_link_to_lidar_tf)
BODY_REAR = 0.0465          # collision box spans [0.0465, 0.6265] m ahead of rear axle
BODY_FRONT = 0.6265
BODY_HALF_W = 0.155


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Stage2DisparityController(Node):
    def __init__(self):
        super().__init__('redbull_stage2_controller')
        defaults = {
            # Speed profile along the race line
            'max_speed': 5.8,
            'tight_speed': 0.9,          # speed where clearance is smallest (cluster, corner)
            'tight_clearance': 0.07,     # clearance (m) treated as "tightest"
            'open_clearance': 0.30,      # clearance (m) above which max_speed is allowed
            'lateral_accel': 5.0,
            'brake_decel': 4.5,
            'accel': 3.8,
            'obstacle_speed': 0.8,
            'min_speed': 0.4,

            'wheelbase': 0.3302,
            'vehicle_width': 0.31,
            'max_steering': 0.4189,

            'latency_comp': 0.12,        # s; steering servo lag compensation
            'ff_gain': 1.0,              # corner-cutting compensation (0 = plain pure pursuit)
            'ff_preview': 0.0,           # where along the look-ahead to sample curvature (0..1)        # s; steering servo lag compensation
            'steer_alpha': 0.85,         # steering smoothing (1.0 = none)
            'min_lookahead': 0.60,
            'max_lookahead': 1.10,
            'lookahead_gain': 0.22,

            # LiDAR guard on the race line
            'guard_margin': -0.02,       # footprint grown (+) / shrunk (-) when testing LiDAR points (m)
            'guard_points': 3,           # LiDAR points inside the footprint to call it blocked
            'guard_cycles': 5,           # consecutive blocked scans before follow-the-gap
            'gap_engage_speed': 0.3,     # only switch to follow-the-gap once slowed below this
            'gap_stuck_time': 2.0,       # s stopped in follow-the-gap before retrying the line
            'clear_time': 0.5,           # s the race line must stay clear before leaving follow-the-gap
            'rejoin_free': 1.2,          # m of free arc needed on the pure-pursuit steering to rejoin
            'rejoin_speed': 1.0,         # speed cap until back on the race line
            'stop_buffer': 0.15,         # stop this far short of an obstacle on the line (m)

            # Follow-the-gap (fallback)
            'disparity_threshold': 0.30,
            'car_inflation_radius': 0.30,
            'gap_fov_deg': 90.0,
            'heading_weight': 1.0,
            'emergency_distance': 0.20,  # bumper-to-obstacle distance that forces min speed

            # Stuck recovery
            'stuck_time': 0.8,
            'reverse_time': 0.8,
            'reverse_speed': -0.6,

            # Startup grace period (the out-lap is not timed, so take it gently)
            'startup_hold': 1.0,         # s of steady scan+odom before the car moves
            'outlap_speed': 4.0,         # speed cap until the cluster has been passed once
            'outlap_end_x': 1.0,         # first time x passes this on the bottom straight -> race pace
            'guard_period': 0.02,        # s between LiDAR guard checks (50 Hz)

            'sensor_timeout': 0.5,
            'control_period': 0.01,
            'waypoint_filename': '',     # optional CSV (x,y[,clearance]) in f1tenth_gym_ros/maps
        }

        for key, value in defaults.items():
            self.declare_parameter(key, value)
        for key in defaults:
            setattr(self, key, self.get_parameter(key).value)

        self.load_race_line()
        self.build_speed_profile()

        self.last_nearest_index = None
        self.scan = None
        self.scan_time = None
        self.odom_time = None
        self.completed_laps = 0
        self.collisions = 0
        self.in_obstacle_zone = False
        self.last_steering = 0.0
        self.last_control = -1.0

        self._angle_key = None
        self._cos = self._sin = self._angles = None
        self._guard_scan_time = None
        self.blocked_count = 0
        self.clear_count = 0
        self.blocked_s = math.inf
        self.gap_since = 0.0
        self.sensors_ok_since = None
        self.outlap = True
        self._guard_time = -1.0
        self.clear_since = None
        self.rejoining = False

        self.stuck_since = None
        self.reverse_until = None
        self.reverse_steer = 0.0
        self.last_cmd_speed = 0.0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.on_odom, 10)

        self.create_timer(0.1, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'Stage 2 Controller v8 (grace period) ready: {len(self.waypoints)} race-line points, max_speed={self.max_speed}')

    # ------------------------------------------------------------------ race line
    def load_race_line(self):
        rows = []
        if self.waypoint_filename:
            from ament_index_python.packages import get_package_share_directory
            maps_dir = Path(get_package_share_directory('f1tenth_gym_ros')) / 'maps'
            with open(maps_dir / str(self.waypoint_filename), newline='') as handle:
                for row in csv.reader(handle):
                    if len(row) >= 2 and row[0].strip() and not row[0].lstrip().startswith('#'):
                        rows.append((float(row[0]), float(row[1]),
                                     float(row[2]) if len(row) >= 3 else 0.3,
                                     float(row[3]) if len(row) >= 4 else 0.0))
        else:
            for line in RACE_LINE.strip().splitlines():
                vals = [float(v) for v in line.split(',')]
                rows.append(tuple(vals) + (0.0,) * (4 - len(vals)))
        if len(rows) < 3:
            raise RuntimeError('Insufficient race-line points')
        self.waypoints = [(r[0], r[1]) for r in rows]
        self.wp = np.array(self.waypoints)
        self.wp_clear = np.array([r[2] for r in rows])
        self.wp_yawoff = np.array([r[3] for r in rows])
        nxt = np.roll(self.wp, -1, axis=0)
        prv = np.roll(self.wp, 1, axis=0)
        t = nxt - prv
        self.wp_yaw = np.arctan2(t[:, 1], t[:, 0])
        self.wp_ds = np.linalg.norm(nxt - self.wp, axis=1)

    def build_speed_profile(self):
        n = len(self.wp)
        # curvature from a wider stencil (smooths DP steps)
        k = 4
        a = np.roll(self.wp, k, axis=0)
        b = self.wp
        c = np.roll(self.wp, -k, axis=0)
        ab = np.linalg.norm(b - a, axis=1)
        bc = np.linalg.norm(c - b, axis=1)
        ca = np.linalg.norm(a - c, axis=1)
        cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        signed = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        curv = 2.0 * cross / np.maximum(ab * bc * ca, 1e-6)
        self.wp_curv = 2.0 * signed / np.maximum(ab * bc * ca, 1e-6)
        self.wp_s = np.concatenate([[0.0], np.cumsum(self.wp_ds)])
        self.track_len = float(self.wp_s[-1])
        v_curve = np.sqrt(self.lateral_accel / np.maximum(curv, 1e-3))

        frac = np.clip((self.wp_clear - self.tight_clearance) /
                       max(1e-3, self.open_clearance - self.tight_clearance), 0.0, 1.0)
        v_clear = self.tight_speed + frac * (self.max_speed - self.tight_speed)
        # the car must already be slow when its footprint reaches a tight spot:
        # take the min over a window covering the car length ahead and behind
        win = 6
        v_clear = np.min(np.stack([np.roll(v_clear, -j) for j in range(-win, win + 1)]), axis=0)

        v = np.minimum(np.minimum(v_curve, v_clear), self.max_speed)
        for _ in range(2):   # braking (backward) and acceleration (forward) passes, loop closed
            for i in range(2 * n, -1, -1):
                j = i % n
                v[j] = min(v[j], math.sqrt(v[(j + 1) % n] ** 2 + 2 * self.brake_decel * self.wp_ds[j]))
            for i in range(2 * n):
                j = i % n
                v[(j + 1) % n] = min(v[(j + 1) % n], math.sqrt(v[j] ** 2 + 2 * self.accel * self.wp_ds[j]))
        self.wp_speed = v

    # ------------------------------------------------------------------ I/O
    def on_lap(self, msg):
        self.completed_laps = int(msg.data)

    def on_collision(self, msg):
        self.collisions = int(msg.data)

    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def drive(self, raw_steering, speed):
        alpha = self.steer_alpha if not self.in_obstacle_zone else 0.5
        smoothed = self.last_steering + alpha * (raw_steering - self.last_steering)
        smoothed = clamp(smoothed, -self.max_steering, self.max_steering)
        self.last_steering = smoothed
        speed = float(clamp(speed, 0.0, self.max_speed))
        self.last_cmd_speed = speed
        self.publish(smoothed, speed)

    def stop(self):
        self.last_steering = 0.0
        self.last_cmd_speed = 0.0
        self.publish(0.0, 0.0)

    def on_scan(self, msg):
        self.scan = msg
        self.scan_time = time.monotonic()

    def fresh(self):
        now = time.monotonic()
        return (
            self.scan is not None
            and self.scan_time is not None
            and self.odom_time is not None
            and (now - self.scan_time) <= self.sensor_timeout
            and (now - self.odom_time) <= self.sensor_timeout
        )

    def watchdog(self):
        if not self.fresh():
            self.stop()

    # ------------------------------------------------------------------ pure pursuit
    def nearest_waypoint(self, x, y):
        n = len(self.wp)
        if self.last_nearest_index is None:
            idx = int(np.argmin((self.wp[:, 0] - x) ** 2 + (self.wp[:, 1] - y) ** 2))
        else:
            cand = (self.last_nearest_index + np.arange(-20, 75)) % n
            d2 = (self.wp[cand, 0] - x) ** 2 + (self.wp[cand, 1] - y) ** 2
            idx = int(cand[np.argmin(d2)])
            if d2.min() > 4.0:
                idx = int(np.argmin((self.wp[:, 0] - x) ** 2 + (self.wp[:, 1] - y) ** 2))
        self.last_nearest_index = idx
        return idx

    def point_at(self, s_arc):
        """Interpolated (x, y, yaw, curvature) at arc length s along the race line."""
        n = len(self.wp)
        s_arc = s_arc % self.track_len
        i = int(np.searchsorted(self.wp_s, s_arc, side='right') - 1) % n
        j = (i + 1) % n
        f = (s_arc - self.wp_s[i]) / max(self.wp_ds[i], 1e-9)
        x = self.wp[i, 0] + f * (self.wp[j, 0] - self.wp[i, 0])
        y = self.wp[i, 1] + f * (self.wp[j, 1] - self.wp[i, 1])
        yaw = math.atan2(self.wp[j, 1] - self.wp[i, 1], self.wp[j, 0] - self.wp[i, 0])
        k = self.wp_curv[i] + f * (self.wp_curv[j] - self.wp_curv[i])
        return x, y, yaw, k

    def pp_steer_to(self, x, y, yaw, tx, ty):
        dx, dy = tx - x, ty - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        d2 = max(0.01, local_x ** 2 + local_y ** 2)
        if local_x <= 0.05:
            return math.copysign(self.max_steering, local_y), local_x, local_y
        return math.atan(self.wheelbase * 2.0 * local_y / d2), local_x, local_y

    def pure_pursuit(self, x, y, yaw, lookahead, v=0.0):
        """Pure pursuit with interpolated look-ahead point and corner-cutting compensation.

        Plain pure pursuit cuts the inside of every curve (it aims at a chord, not the arc).
        We evaluate what pure pursuit would command if the car were exactly on the line
        (`ideal`) and compare it with the steering the line's curvature really needs (`ff`);
        the difference is pure pursuit's built-in cutting error and is added back.
        Returns (steering, target_idx, local_x, local_y, nearest_idx)."""
        n = len(self.wp)
        near = self.nearest_waypoint(x, y)
        best = None
        for a in ((near - 1) % n, near):
            b = (a + 1) % n
            ax, ay = self.wp[a]
            sx, sy = self.wp[b, 0] - ax, self.wp[b, 1] - ay
            seg2 = sx * sx + sy * sy
            t = clamp(((x - ax) * sx + (y - ay) * sy) / max(seg2, 1e-9), 0.0, 1.0)
            d2 = (x - ax - t * sx) ** 2 + (y - ay - t * sy) ** 2
            if best is None or d2 < best[0]:
                best = (d2, a, t)
        _, seg, t = best
        s_proj = self.wp_s[seg] + t * self.wp_ds[seg]

        tx, ty, _, _ = self.point_at(s_proj + lookahead)
        steer, local_x, local_y = self.pp_steer_to(x, y, yaw, tx, ty)

        if self.ff_gain > 0.0:
            ox, oy, oyaw, _ = self.point_at(s_proj)
            ideal, _, _ = self.pp_steer_to(ox, oy, oyaw, tx, ty)
            _, _, _, k_ff = self.point_at(s_proj + self.ff_preview * lookahead)
            ff = math.atan(self.wheelbase * k_ff)
            steer += self.ff_gain * (ff - ideal)

        steer = clamp(steer, -self.max_steering, self.max_steering)
        target_idx = int(np.searchsorted(self.wp_s, (s_proj + lookahead) % self.track_len)) % n
        return steer, target_idx, local_x, local_y, near

    # ------------------------------------------------------------------ LiDAR
    def scan_arrays(self, msg):
        n = len(msg.ranges)
        key = (n, msg.angle_min, msg.angle_increment)
        if key != self._angle_key:
            self._angles = msg.angle_min + np.arange(n) * msg.angle_increment
            self._cos = np.cos(self._angles)
            self._sin = np.sin(self._angles)
            self._angle_key = key
        r = np.asarray(msg.ranges, dtype=float)
        valid = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        return r, valid

    def guard_race_line(self, x, y, yaw, near_idx, v, r, valid):
        """Distance along the race line to the first footprint pose that contains LiDAR points
        (inf if the next `horizon` metres are clear)."""
        horizon = max(2.0, 1.2 * v + v * v / (2.0 * self.brake_decel) + 0.8)
        sel = valid & (r < horizon + 1.5)
        if not np.any(sel):
            return math.inf
        lx = LIDAR_X + r[sel] * self._cos[sel]
        ly = r[sel] * self._sin[sel]
        c, s = math.cos(yaw), math.sin(yaw)
        px = x + c * lx - s * ly          # LiDAR points in the map frame
        py = y + s * lx + c * ly

        n = len(self.wp)
        steps = []
        dist = 0.0
        i = near_idx
        while dist < horizon and len(steps) < 60:
            steps.append((i, dist))
            dist += self.wp_ds[i]
            i = (i + 1) % n
        idxs = np.array([k for k, _ in steps])
        dists = np.array([d for _, d in steps])

        ahead = lx > BODY_FRONT + 0.05            # ignore what is beside the car right now
        if not np.any(ahead):
            return math.inf
        px, py = px[ahead], py[ahead]
        heading = self.wp_yaw[idxs] + self.wp_yawoff[idxs]
        cy, sy = np.cos(heading), np.sin(heading)
        dx = px[None, :] - self.wp[idxs, 0][:, None]
        dy = py[None, :] - self.wp[idxs, 1][:, None]
        along = cy[:, None] * dx + sy[:, None] * dy
        across = -sy[:, None] * dx + cy[:, None] * dy
        m = self.guard_margin
        inside = ((along > BODY_REAR - m) & (along < BODY_FRONT + m)
                  & (np.abs(across) < BODY_HALF_W + m))
        counts = inside.sum(axis=1)
        hit = np.nonzero(counts >= self.guard_points)[0]
        if len(hit) == 0:
            return math.inf
        return float(dists[hit[0]])

    def disparity_extender_steering(self, r, valid, preferred_angle):
        """Follow-the-gap: extend disparities by the car half-width, pick the deepest opening
        closest to the race-line direction, steer toward it with pure-pursuit geometry."""
        fov = math.radians(self.gap_fov_deg)
        sel = np.abs(self._angles) <= fov
        ang = self._angles[sel]
        rng = np.minimum(np.where(valid[sel], r[sel], 5.0), 5.0)
        n = len(rng)
        if n < 10:
            return 0.0, 0.0

        inc = abs(self.scan.angle_increment)
        extended = rng.copy()
        diffs = np.diff(rng)
        for i in np.nonzero(np.abs(diffs) >= self.disparity_threshold)[0]:
            closer = min(rng[i], rng[i + 1])
            k = int(math.ceil(math.atan2(self.car_inflation_radius, max(0.15, closer)) / inc))
            if diffs[i] > 0:
                lo, hi = max(0, i - 2), min(n, i + k + 1)
            else:
                lo, hi = max(0, i + 1 - k), min(n, i + 4)
            extended[lo:hi] = np.minimum(extended[lo:hi], closer)

        smoothed = np.convolve(extended, np.ones(7) / 7.0, mode='same')
        score = np.minimum(smoothed, 3.0) - self.heading_weight * np.abs(ang - preferred_angle)
        best = int(np.argmax(score))
        a = float(ang[best])
        d = clamp(float(extended[best]), 0.5, 1.5)
        tx = LIDAR_X + d * math.cos(a)
        ty = d * math.sin(a)
        steer = math.atan(2.0 * self.wheelbase * ty / (tx * tx + ty * ty))
        return clamp(steer, -self.max_steering, self.max_steering), float(extended[best])

    def arc_free_distance(self, steer, r, valid, max_dist=1.5, margin=0.03):
        """Distance the car can drive on a constant-steering arc before its footprint
        touches a LiDAR point (car frame, rear axle origin)."""
        m = valid & (r < max_dist + 1.0)
        if not np.any(m):
            return max_dist
        qx = LIDAR_X + r[m] * self._cos[m]
        qy = r[m] * self._sin[m]
        k = math.tan(steer) / self.wheelbase
        s = np.arange(0.05, max_dist + 1e-6, 0.05)
        if abs(k) < 1e-4:
            ox, oy, oh = s, np.zeros_like(s), np.zeros_like(s)
        else:
            oh = k * s
            ox = np.sin(oh) / k
            oy = (1.0 - np.cos(oh)) / k
        dx = qx[None, :] - ox[:, None]
        dy = qy[None, :] - oy[:, None]
        c, sn = np.cos(oh)[:, None], np.sin(oh)[:, None]
        along = c * dx + sn * dy
        across = -sn * dx + c * dy
        hit = ((along > BODY_REAR - margin) & (along < BODY_FRONT + margin)
               & (np.abs(across) < BODY_HALF_W + margin)).any(axis=1)
        idx = np.nonzero(hit)[0]
        return float(s[idx[0]] - 0.05) if len(idx) else max_dist

    def bumper_distance(self, r, valid):
        """Free distance in front of the front bumper, inside the car's width."""
        lx = LIDAR_X + r * self._cos
        ly = r * self._sin
        m = valid & (lx > BODY_FRONT - 0.05) & (np.abs(ly) < BODY_HALF_W + 0.03)
        if not np.any(m):
            return math.inf
        return float(np.min(lx[m])) - BODY_FRONT

    # ------------------------------------------------------------------ recovery
    def update_stuck(self, vx, bumper, r, valid, now):
        if self.reverse_until is not None:
            if now < self.reverse_until:
                return True
            self.reverse_until = None
            self.stuck_since = None
            self.last_steering = 0.0
        if self.last_cmd_speed > 0.2 and abs(vx) < 0.05:
            if self.stuck_since is None:
                self.stuck_since = now
            elif now - self.stuck_since > self.stuck_time and bumper < 0.3:
                left = valid & (self._angles > 0.0) & (self._angles < math.radians(90))
                right = valid & (self._angles < 0.0) & (self._angles > math.radians(-90))
                l_min = float(np.min(r[left])) if np.any(left) else 10.0
                r_min = float(np.min(r[right])) if np.any(right) else 10.0
                self.reverse_steer = self.max_steering if l_min < r_min else -self.max_steering
                self.reverse_until = now + self.reverse_time
                self.get_logger().warn('Stuck - reversing')
                return True
        else:
            self.stuck_since = None
        return False

    # ------------------------------------------------------------------ main loop
    def on_odom(self, msg):
        self.odom_time = time.monotonic()
        if not self.fresh():
            self.sensors_ok_since = None
            self.stop()
            return
        now = self.odom_time
        if now - self.last_control < self.control_period - 1e-4:
            return
        self.last_control = now

        # Startup grace: wait for steady sensors before moving
        if self.sensors_ok_since is None:
            self.sensors_ok_since = now
        if now - self.sensors_ok_since < self.startup_hold:
            self.stop()
            return

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        vx = msg.twist.twist.linear.x
        v = max(0.0, vx)

        r, valid = self.scan_arrays(self.scan)
        bumper = self.bumper_distance(r, valid)

        if self.update_stuck(vx, bumper, r, valid, now):
            self.blocked_count = self.clear_count = 0
            self.last_cmd_speed = 0.0
            self.publish(self.reverse_steer, self.reverse_speed)
            return

        lookahead = clamp(self.min_lookahead + self.lookahead_gain * v,
                          self.min_lookahead, self.max_lookahead)
        # compensate steering-actuator lag: run pure pursuit from the pose ~latency s ahead
        tau = self.latency_comp
        yaw_rate = v * math.tan(self.last_steering) / self.wheelbase
        px = pos.x + v * tau * math.cos(yaw + 0.5 * yaw_rate * tau)
        py = pos.y + v * tau * math.sin(yaw + 0.5 * yaw_rate * tau)
        pyaw = yaw + yaw_rate * tau
        pp_steer, target_idx, lx, ly, near_idx = self.pure_pursuit(px, py, pyaw, lookahead, v)

        # LiDAR guard on the race line (once per new scan)
        if self.scan_time != self._guard_scan_time and now - self._guard_time >= self.guard_period:
            self._guard_scan_time = self.scan_time
            self._guard_time = now
            self.blocked_s = self.guard_race_line(pos.x, pos.y, yaw, near_idx, v, r, valid)
            if self.blocked_s < math.inf:
                self.blocked_count += 1
                self.clear_count = 0
            else:
                self.blocked_count = 0
                self.clear_count += 1
            if (not self.in_obstacle_zone and self.blocked_count >= self.guard_cycles
                    and (v < self.gap_engage_speed or self.blocked_s < 0.5)):
                self.in_obstacle_zone = True
                self.gap_since = now
                self.get_logger().warn('Race line blocked - follow the gap')
            if self.blocked_s < math.inf:
                self.clear_since = None
            elif self.clear_since is None:
                self.clear_since = now
            if (self.in_obstacle_zone and self.clear_since is not None
                    and now - self.clear_since > self.clear_time
                    and self.arc_free_distance(pp_steer, r, valid) >= self.rejoin_free):
                self.in_obstacle_zone = False
                self.rejoining = True
                self.get_logger().info('Race line clear - pure pursuit')

        if self.in_obstacle_zone:
            if v > 0.05:
                self.gap_since = now
            elif now - self.gap_since > self.gap_stuck_time:
                # stalled in front of something: back out, then look for a gap again
                left = valid & (self._angles > 0.0) & (self._angles < math.radians(90))
                right = valid & (self._angles < 0.0) & (self._angles > math.radians(-90))
                l_min = float(np.min(r[left])) if np.any(left) else 10.0
                r_min = float(np.min(r[right])) if np.any(right) else 10.0
                self.reverse_steer = self.max_steering if l_min < r_min else -self.max_steering
                self.reverse_until = now + self.reverse_time
                self.gap_since = now + self.reverse_time
                self.get_logger().warn('Follow the gap stalled - backing out')
                self.publish(self.reverse_steer, self.reverse_speed)
                return

        if not self.in_obstacle_zone:
            steering = pp_steer
            # look a little ahead in the speed profile so braking starts early enough
            target_speed = float(np.min(self.wp_speed[(near_idx + np.arange(0, 4)) % len(self.wp)]))
            if self.rejoining:
                wx, wy = self.wp[near_idx]
                if math.hypot(pos.x - wx, pos.y - wy) < 0.10:
                    self.rejoining = False
                else:
                    target_speed = min(target_speed, self.rejoin_speed)
                    free = self.arc_free_distance(pp_steer, r, valid)
                    target_speed = min(target_speed,
                                       math.sqrt(2.0 * self.brake_decel * max(0.0, free - 0.05)))
            # never drive faster than we can stop before a LiDAR return on our line
            if self.blocked_s < math.inf:
                stop_room = max(0.0, self.blocked_s - self.stop_buffer)
                target_speed = min(target_speed, math.sqrt(2.0 * self.brake_decel * stop_room))
        else:
            steering, gap_dist = self.disparity_extender_steering(r, valid, math.atan2(ly, lx))
            free = self.arc_free_distance(steering, r, valid)
            if free < 0.15 and bumper < 0.35:
                # too close to turn away: back up with the nose swinging toward the gap
                self.last_cmd_speed = 0.0
                self.last_steering = 0.0
                self.publish(-steering, self.reverse_speed)
                return
            target_speed = self.obstacle_speed * (1.0 - 0.4 * abs(steering) / self.max_steering)
            if gap_dist < 1.0:
                target_speed = min(target_speed, 0.5)
            target_speed = min(target_speed,
                               math.sqrt(2.0 * self.brake_decel * max(0.0, free - 0.05)))
            target_speed = max(target_speed, 0.25) if free > 0.15 else target_speed

        # Out-lap: race pace only after the first pass through the cluster
        if self.outlap:
            if pos.x > self.outlap_end_x and abs(pos.y) < 1.5:
                self.outlap = False
                self.get_logger().info('Out-lap done - race pace')
            else:
                target_speed = min(target_speed, self.outlap_speed)

        self.drive(steering, target_speed)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = Stage2DisparityController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
