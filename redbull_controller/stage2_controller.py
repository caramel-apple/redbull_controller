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
0.0341,-0.4225,0.2022,-0.0543
0.1522,-0.3722,0.1733,-0.0921
0.2660,-0.3082,0.1242,-0.1097
0.3835,-0.1969,0.0619,-0.1090
0.5089,-0.1462,0.0755,-0.0887
0.6309,-0.0805,0.0693,-0.0395
0.7610,-0.0428,0.0664,0.0071
0.8905,-0.0212,0.0624,0.0396
1.0217,-0.0080,0.0654,0.0703
1.1500,-0.0143,0.0634,0.0769
1.2754,-0.0200,0.0659,0.0713
1.4008,-0.0266,0.0701,0.0520
1.5261,-0.0331,0.0655,0.0272
1.6514,-0.0388,0.0608,0.0137
1.7766,-0.0503,0.0560,0.0183
1.9011,-0.0708,0.0557,0.0322
2.0238,-0.1043,0.0612,0.0536
2.1468,-0.1557,0.0875,0.0754
2.2697,-0.1969,0.1666,0.0822
2.3939,-0.2497,0.2139,0.0742
2.5217,-0.2814,0.2473,0.0544
2.6494,-0.3053,0.2749,0.0265
2.7793,-0.3165,0.3184,-0.0054
2.9076,-0.3098,0.3819,-0.0323
3.0349,-0.2986,0.4647,-0.0577
3.1611,-0.2741,0.5512,-0.0742
3.2856,-0.2468,0.5668,-0.0826
3.4101,-0.2158,0.5851,-0.0844
3.5343,-0.1826,0.6078,-0.0785
3.6591,-0.1535,0.6218,-0.0665
3.7842,-0.1250,0.6076,-0.0485
3.9103,-0.1010,0.5959,-0.0300
4.0366,-0.0818,0.5803,-0.0112
4.1629,-0.0650,0.5656,0.0065
4.2896,-0.0508,0.5555,0.0194
4.4160,-0.0389,0.5369,0.0273
4.5425,-0.0264,0.5269,0.0315
4.6686,-0.0153,0.5032,0.0324
4.7948,-0.0012,0.4907,0.0302
4.9211,0.0124,0.4710,0.0261
5.0470,0.0276,0.4492,0.0222
5.1731,0.0450,0.4299,0.0181
5.2988,0.0631,0.4073,0.0129
5.4245,0.0829,0.3858,0.0092
5.5500,0.1043,0.3629,0.0047
5.6754,0.1269,0.3418,-0.0004
5.8006,0.1498,0.3130,-0.0040
5.9256,0.1752,0.2737,-0.0077
6.0512,0.1995,0.2399,-0.0099
6.1765,0.2239,0.2197,-0.0115
6.3023,0.2481,0.2065,-0.0122
6.4278,0.2683,0.2070,-0.0119
6.5536,0.2910,0.1902,-0.0117
6.6801,0.3082,0.1864,-0.0129
6.8061,0.3245,0.1790,-0.0125
6.9322,0.3384,0.1701,-0.0117
7.0585,0.3499,0.1694,-0.0126
7.1846,0.3581,0.1665,-0.0127
7.3107,0.3645,0.1686,-0.0133
7.4372,0.3678,0.1728,-0.0152
7.5635,0.3687,0.1695,-0.0179
7.6900,0.3653,0.1701,-0.0198
7.8169,0.3579,0.1566,-0.0209
7.9440,0.3449,0.1396,-0.0187
8.0705,0.3249,0.1282,-0.0110
8.1960,0.2970,0.1011,0.0039
8.3203,0.2582,0.0844,0.0255
8.4438,0.2097,0.0760,0.0493
8.5663,0.1507,0.0707,0.0788
8.6916,0.0862,0.0697,0.0996
8.8183,0.0268,0.0631,0.1015
8.9565,-0.0403,0.0726,0.0912
9.0942,-0.0674,0.1462,0.0690
9.2359,-0.0847,0.1925,0.0371
9.3732,-0.0687,0.1656,0.0061
9.5093,-0.0359,0.1076,-0.0136
9.6369,0.0227,0.0848,-0.0289
9.7563,0.0987,0.0697,-0.0328
9.8640,0.1935,0.0708,-0.0338
9.9556,0.3046,0.0797,-0.0275
10.0297,0.4241,0.1067,-0.0198
10.0809,0.5508,0.1400,-0.0116
10.1160,0.6770,0.1496,-0.0067
10.1278,0.8047,0.1778,-0.0112
10.1267,0.9288,0.1881,-0.0227
10.1049,1.0513,0.1887,-0.0357
10.0695,1.1765,0.2077,-0.0486
10.0263,1.3038,0.2406,-0.0569
9.9812,1.4333,0.2784,-0.0493
9.9378,1.5668,0.3193,-0.0289
9.9039,1.7060,0.3575,0.0053
9.8835,1.8476,0.3742,0.0485
9.8768,1.9905,0.3749,0.0919
9.8861,2.1302,0.3752,0.1294
9.9010,2.2666,0.3809,0.1532
9.9219,2.4018,0.3842,0.1596
9.9444,2.5349,0.3834,0.1542
9.9625,2.6683,0.3860,0.1397
9.9806,2.8019,0.3883,0.1191
9.9955,2.9356,0.3893,0.1001
10.0068,3.0693,0.3924,0.0835
10.0174,3.2029,0.3947,0.0678
10.0257,3.3363,0.3931,0.0545
10.0338,3.4698,0.3960,0.0428
10.0390,3.6036,0.4011,0.0317
10.0430,3.7372,0.4049,0.0212
10.0452,3.8708,0.4084,0.0113
10.0463,4.0040,0.4121,0.0027
10.0463,4.1371,0.4154,-0.0051
10.0446,4.2703,0.4194,-0.0118
10.0419,4.4033,0.4229,-0.0178
10.0377,4.5361,0.4275,-0.0231
10.0331,4.6686,0.4315,-0.0271
10.0273,4.8013,0.4356,-0.0293
10.0205,4.9336,0.4402,-0.0303
10.0132,5.0656,0.4444,-0.0299
10.0054,5.1975,0.4482,-0.0282
9.9972,5.3292,0.4526,-0.0257
9.9887,5.4609,0.4565,-0.0227
9.9789,5.5922,0.4611,-0.0201
9.9690,5.7231,0.4657,-0.0184
9.9584,5.8543,0.4700,-0.0168
9.9466,5.9851,0.4757,-0.0144
9.9346,6.1155,0.4810,-0.0121
9.9217,6.2455,0.4879,-0.0078
9.9106,6.3754,0.4925,-0.0002
9.8977,6.5048,0.4994,0.0097
9.8878,6.6335,0.5041,0.0228
9.8768,6.7613,0.5089,0.0370
9.8663,6.8875,0.5147,0.0543
9.8556,7.0122,0.5191,0.0701
9.8400,7.1339,0.5303,0.0844
9.8257,7.2537,0.5361,0.0951
9.8006,7.3691,0.5104,0.1033
9.7750,7.4810,0.4520,0.1068
9.7383,7.5866,0.4473,0.1049
9.6993,7.6870,0.4087,0.0986
9.6505,7.7794,0.3998,0.0894
9.5970,7.8641,0.3757,0.0773
9.5371,7.9392,0.3745,0.0616
9.4750,8.0068,0.3515,0.0483
9.4074,8.0650,0.3521,0.0379
9.3381,8.1179,0.3407,0.0262
9.2629,8.1634,0.3439,0.0159
9.1850,8.2054,0.3462,0.0048
9.1003,8.2381,0.3510,-0.0087
9.0133,8.2677,0.3623,-0.0212
8.9198,8.2862,0.3725,-0.0343
8.8224,8.3020,0.3896,-0.0456
8.7228,8.3101,0.4170,-0.0530
8.6188,8.3113,0.4386,-0.0585
8.5122,8.3090,0.4555,-0.0641
8.4040,8.3015,0.4501,-0.0660
8.2924,8.2895,0.4469,-0.0652
8.1794,8.2759,0.4256,-0.0620
8.0622,8.2580,0.4271,-0.0551
7.9440,8.2433,0.4042,-0.0438
7.8242,8.2257,0.4031,-0.0282
7.7011,8.2116,0.3906,-0.0118
7.5781,8.1992,0.3864,0.0070
7.4557,8.1899,0.3699,0.0265
7.3316,8.1815,0.3703,0.0437
7.2086,8.1778,0.3493,0.0596
7.0845,8.1724,0.3504,0.0733
6.9604,8.1715,0.3316,0.0824
6.8367,8.1685,0.3303,0.0880
6.7130,8.1682,0.3060,0.0891
6.5890,8.1644,0.3036,0.0848
6.4640,8.1622,0.2798,0.0775
6.3392,8.1564,0.2759,0.0675
6.2126,8.1515,0.2487,0.0564
6.0852,8.1457,0.1674,0.0461
5.9569,8.1414,0.1486,0.0385
5.8278,8.1386,0.1431,0.0342
5.6978,8.1400,0.1441,0.0352
5.5677,8.1452,0.1472,0.0398
5.4372,8.1566,0.1530,0.0487
5.3076,8.1738,0.1624,0.0607
5.1785,8.1958,0.1872,0.0736
5.0504,8.2246,0.2230,0.0862
4.9229,8.2559,0.2548,0.0952
4.7959,8.2911,0.2769,0.0995
4.6684,8.3271,0.2979,0.0975
4.5404,8.3593,0.3167,0.0885
4.4114,8.3914,0.3381,0.0728
4.2814,8.4161,0.3555,0.0530
4.1512,8.4379,0.3730,0.0288
4.0202,8.4530,0.3865,0.0041
3.8891,8.4635,0.3993,-0.0189
3.7589,8.4679,0.4069,-0.0394
3.6290,8.4710,0.4153,-0.0542
3.4997,8.4694,0.4200,-0.0639
3.3711,8.4684,0.4260,-0.0685
3.2425,8.4651,0.4304,-0.0692
3.1144,8.4620,0.4348,-0.0676
2.9864,8.4589,0.4397,-0.0659
2.8586,8.4556,0.4440,-0.0622
2.7309,8.4520,0.4484,-0.0592
2.6038,8.4474,0.4515,-0.0559
2.4765,8.4439,0.4562,-0.0528
2.3497,8.4387,0.4586,-0.0485
2.2229,8.4353,0.4621,-0.0443
2.0964,8.4294,0.4646,-0.0400
1.9700,8.4261,0.4644,-0.0352
1.8438,8.4205,0.4674,-0.0297
1.7176,8.4169,0.4643,-0.0246
1.5918,8.4121,0.4670,-0.0200
1.4660,8.4087,0.4642,-0.0152
1.3405,8.4037,0.4667,-0.0111
1.2152,8.4000,0.4636,-0.0081
1.0899,8.3952,0.4660,-0.0052
0.9648,8.3919,0.4602,-0.0012
0.8399,8.3865,0.4633,0.0029
0.7145,8.3833,0.4625,0.0071
0.5898,8.3795,0.4632,0.0117
0.4650,8.3771,0.4609,0.0175
0.3403,8.3741,0.4612,0.0212
0.2152,8.3718,0.4630,0.0238
0.0910,8.3707,0.4566,0.0264
-0.0332,8.3678,0.4588,0.0296
-0.1579,8.3665,0.4578,0.0305
-0.2827,8.3648,0.4606,0.0315
-0.4067,8.3649,0.4533,0.0340
-0.5309,8.3625,0.4564,0.0343
-0.6558,8.3627,0.4586,0.0336
-0.7798,8.3624,0.4569,0.0333
-0.9042,8.3618,0.4580,0.0338
-1.0284,8.3622,0.4562,0.0315
-1.1533,8.3622,0.4598,0.0302
-1.2774,8.3646,0.4540,0.0295
-1.4019,8.3644,0.4598,0.0286
-1.5268,8.3682,0.4599,0.0277
-1.6513,8.3703,0.4643,0.0287
-1.7760,8.3751,0.4628,0.0316
-1.9008,8.3791,0.4636,0.0330
-2.0258,8.3852,0.4643,0.0341
-2.1501,8.3914,0.4629,0.0356
-2.2746,8.3957,0.4626,0.0358
-2.3994,8.3999,0.4609,0.0325
-2.5240,8.4036,0.4515,0.0312
-2.6484,8.4047,0.4391,0.0307
-2.7734,8.4033,0.4300,0.0288
-2.8984,8.4023,0.4099,0.0287
-3.0235,8.3976,0.3858,0.0303
-3.1487,8.3920,0.3659,0.0314
-3.2741,8.3842,0.3411,0.0315
-3.3998,8.3741,0.3193,0.0304
-3.5256,8.3624,0.3042,0.0256
-3.6516,8.3474,0.2933,0.0180
-3.7778,8.3300,0.2764,0.0073
-3.9043,8.3090,0.2535,-0.0054
-4.0309,8.2867,0.2366,-0.0184
-4.1575,8.2624,0.2199,-0.0298
-4.2845,8.2390,0.1968,-0.0382
-4.4118,8.2168,0.1554,-0.0433
-4.5394,8.1995,0.1278,-0.0441
-4.6671,8.1876,0.1154,-0.0396
-4.7947,8.1824,0.1087,-0.0300
-4.9220,8.1853,0.1061,-0.0161
-5.0487,8.1972,0.1075,0.0006
-5.1742,8.2173,0.1134,0.0176
-5.2989,8.2455,0.1242,0.0330
-5.4228,8.2792,0.1936,0.0460
-5.5464,8.3173,0.2745,0.0527
-5.6700,8.3579,0.3060,0.0544
-5.7946,8.4002,0.3419,0.0523
-5.9201,8.4382,0.3755,0.0428
-6.0466,8.4758,0.4110,0.0268
-6.1753,8.5075,0.4441,0.0074
-6.3041,8.5330,0.4730,-0.0145
-6.4335,8.5541,0.4993,-0.0361
-6.5628,8.5693,0.5214,-0.0535
-6.6916,8.5826,0.5421,-0.0653
-6.8204,8.5941,0.5614,-0.0706
-6.9489,8.6053,0.5800,-0.0697
-7.0772,8.6184,0.5996,-0.0641
-7.2055,8.6325,0.6179,-0.0566
-7.3337,8.6502,0.5918,-0.0478
-7.4620,8.6698,0.5695,-0.0389
-7.5899,8.6902,0.5323,-0.0313
-7.7179,8.7152,0.5105,-0.0243
-7.8457,8.7398,0.4734,-0.0161
-7.9735,8.7685,0.4476,-0.0075
-8.1011,8.7978,0.4114,0.0018
-8.2287,8.8301,0.3816,0.0117
-8.3563,8.8634,0.3479,0.0205
-8.4842,8.8981,0.3214,0.0281
-8.6120,8.9318,0.2919,0.0335
-8.7403,8.9648,0.2087,0.0347
-8.8690,8.9948,0.1647,0.0318
-8.9979,9.0231,0.1554,0.0280
-9.1271,9.0450,0.1556,0.0208
-9.2566,9.0636,0.1579,0.0101
-9.3864,9.0784,0.1623,0.0001
-9.5153,9.0868,0.1686,-0.0093
-9.6444,9.0917,0.1800,-0.0193
-9.7734,9.0933,0.1918,-0.0262
-9.9019,9.0915,0.1919,-0.0269
-10.0302,9.0890,0.1920,-0.0241
-10.1585,9.0850,0.1896,-0.0166
-10.2870,9.0833,0.1837,-0.0048
-10.4152,9.0788,0.1835,0.0082
-10.5433,9.0767,0.1794,0.0207
-10.6700,9.0711,0.1715,0.0301
-10.7972,9.0646,0.1616,0.0383
-10.9229,9.0553,0.1502,0.0408
-11.0485,9.0393,0.1547,0.0383
-11.1724,9.0229,0.1579,0.0318
-11.2946,8.9958,0.1625,0.0220
-11.4168,8.9670,0.1730,0.0077
-11.5371,8.9305,0.2030,-0.0054
-11.6571,8.8906,0.2819,-0.0146
-11.7769,8.8461,0.3366,-0.0200
-11.8976,8.8012,0.3745,-0.0177
-12.0204,8.7538,0.4166,-0.0082
-12.1445,8.7092,0.4570,0.0084
-12.2689,8.6642,0.4998,0.0296
-12.3946,8.6230,0.5405,0.0543
-12.5183,8.5831,0.5615,0.0803
-12.6413,8.5442,0.5473,0.1053
-12.7614,8.5057,0.5368,0.1254
-12.8790,8.4637,0.5337,0.1402
-12.9932,8.4185,0.5298,0.1481
-13.1028,8.3661,0.5317,0.1493
-13.2080,8.3061,0.5375,0.1440
-13.3066,8.2369,0.5479,0.1338
-13.3974,8.1588,0.5623,0.1193
-13.4789,8.0713,0.5325,0.1014
-13.5507,7.9751,0.4900,0.0824
-13.6113,7.8722,0.4566,0.0635
-13.6606,7.7650,0.4331,0.0448
-13.6995,7.6545,0.4159,0.0276
-13.7292,7.5420,0.4109,0.0136
-13.7488,7.4264,0.4056,-0.0005
-13.7619,7.3077,0.4028,-0.0131
-13.7686,7.1885,0.4069,-0.0227
-13.7674,7.0660,0.4128,-0.0292
-13.7640,6.9417,0.4108,-0.0339
-13.7547,6.8152,0.4106,-0.0340
-13.7452,6.6871,0.4089,-0.0302
-13.7324,6.5569,0.4127,-0.0239
-13.7212,6.4245,0.4168,-0.0156
-13.7098,6.2919,0.4231,-0.0047
-13.6999,6.1587,0.4292,0.0072
-13.6906,6.0238,0.4379,0.0178
-13.6853,5.8895,0.4440,0.0298
-13.6799,5.7559,0.4515,0.0417
-13.6772,5.6205,0.4587,0.0516
-13.6771,5.4858,0.4649,0.0608
-13.6781,5.3514,0.4708,0.0699
-13.6809,5.2166,0.4763,0.0763
-13.6855,5.0823,0.4807,0.0814
-13.6909,4.9482,0.4872,0.0855
-13.6975,4.8140,0.5198,0.0884
-13.7050,4.6801,0.5809,0.0898
-13.7130,4.5461,0.6628,0.0892
-13.7223,4.4127,0.6459,0.0877
-13.7306,4.2798,0.5767,0.0854
-13.7392,4.1464,0.5315,0.0808
-13.7482,4.0132,0.5158,0.0757
-13.7569,3.8805,0.5165,0.0707
-13.7647,3.7477,0.5163,0.0648
-13.7731,3.6151,0.5180,0.0581
-13.7801,3.4827,0.5170,0.0518
-13.7879,3.3505,0.5199,0.0450
-13.7937,3.2183,0.5201,0.0386
-13.8002,3.0864,0.5231,0.0312
-13.8046,2.9548,0.5216,0.0225
-13.8100,2.8232,0.5268,0.0153
-13.8115,2.6917,0.5345,0.0080
-13.8128,2.5607,0.5356,0.0010
-13.8140,2.4298,0.5410,-0.0036
-13.8123,2.2990,0.5448,-0.0042
-13.8111,2.1685,0.5465,-0.0042
-13.8077,2.0384,0.5497,-0.0023
-13.8053,1.9082,0.5503,0.0025
-13.7995,1.7780,0.5525,0.0089
-13.7928,1.6485,0.5552,0.0167
-13.7841,1.5198,0.5591,0.0267
-13.7733,1.3925,0.5641,0.0393
-13.7597,1.2667,0.5681,0.0510
-13.7430,1.1432,0.5312,0.0618
-13.7223,1.0221,0.4907,0.0712
-13.6957,0.9045,0.4723,0.0766
-13.6643,0.7907,0.4614,0.0770
-13.6273,0.6811,0.4479,0.0748
-13.5825,0.5769,0.4409,0.0709
-13.5309,0.4792,0.4381,0.0628
-13.4728,0.3880,0.4399,0.0524
-13.4089,0.3035,0.4441,0.0408
-13.3376,0.2271,0.4501,0.0281
-13.2619,0.1569,0.4607,0.0135
-13.1797,0.0950,0.4735,0.0011
-13.0931,0.0391,0.4919,-0.0110
-13.0005,-0.0085,0.5159,-0.0236
-12.9048,-0.0514,0.5388,-0.0350
-12.8043,-0.0848,0.5095,-0.0448
-12.7000,-0.1132,0.4923,-0.0539
-12.5920,-0.1350,0.4676,-0.0618
-12.4818,-0.1528,0.4586,-0.0650
-12.3696,-0.1648,0.4462,-0.0654
-12.2543,-0.1735,0.4354,-0.0635
-12.1369,-0.1798,0.4261,-0.0597
-12.0177,-0.1844,0.4169,-0.0525
-11.8977,-0.1881,0.4124,-0.0433
-11.7761,-0.1907,0.4022,-0.0326
-11.6532,-0.1941,0.3960,-0.0210
-11.5299,-0.1975,0.3867,-0.0086
-11.4065,-0.2024,0.3839,0.0060
-11.2828,-0.2071,0.3770,0.0194
-11.1567,-0.2132,0.3667,0.0316
-11.0330,-0.2218,0.3680,0.0460
-10.9091,-0.2287,0.3643,0.0600
-10.7818,-0.2385,0.3568,0.0729
-10.6566,-0.2504,0.3577,0.0856
-10.5302,-0.2612,0.3525,0.0959
-10.4048,-0.2756,0.3512,0.1028
-10.2796,-0.2866,0.3520,0.1053
-10.1511,-0.2989,0.3534,0.1039
-10.0242,-0.3108,0.3541,0.0979
-9.8947,-0.3200,0.3600,0.0925
-9.7654,-0.3318,0.3624,0.0840
-9.6333,-0.3400,0.3457,0.0770
-9.5014,-0.3562,0.3511,0.0722
-9.3694,-0.3683,0.3174,0.0685
-9.2364,-0.3911,0.3107,0.0685
-9.1054,-0.4142,0.2781,0.0708
-8.9735,-0.4440,0.2428,0.0750
-8.8435,-0.4801,0.1497,0.0786
-8.7139,-0.5164,0.0861,0.0816
-8.5847,-0.5586,0.0764,0.0801
-8.4552,-0.5971,0.0749,0.0736
-8.3254,-0.6334,0.0739,0.0620
-8.1950,-0.6657,0.0743,0.0472
-8.0639,-0.6892,0.0763,0.0308
-7.9328,-0.7083,0.0941,0.0127
-7.8019,-0.7200,0.0980,-0.0020
-7.6716,-0.7258,0.1000,-0.0131
-7.5421,-0.7285,0.1013,-0.0200
-7.4129,-0.7285,0.1035,-0.0212
-7.2843,-0.7273,0.1061,-0.0174
-7.1560,-0.7267,0.1081,-0.0095
-7.0282,-0.7246,0.1069,-0.0001
-6.9005,-0.7234,0.1035,0.0085
-6.7733,-0.7194,0.1014,0.0144
-6.6465,-0.7118,0.0924,0.0154
-6.5200,-0.7003,0.0833,0.0110
-6.3944,-0.6816,0.0807,0.0027
-6.2701,-0.6572,0.0823,-0.0088
-6.1463,-0.6263,0.0846,-0.0203
-6.0233,-0.5902,0.1024,-0.0310
-5.9005,-0.5518,0.1856,-0.0380
-5.7778,-0.5123,0.2571,-0.0414
-5.6543,-0.4737,0.2936,-0.0392
-5.5302,-0.4402,0.3261,-0.0329
-5.4050,-0.4095,0.3596,-0.0200
-5.2793,-0.3890,0.3845,-0.0034
-5.1533,-0.3711,0.4102,0.0154
-5.0273,-0.3654,0.4250,0.0341
-4.9019,-0.3619,0.4387,0.0509
-4.7767,-0.3664,0.4447,0.0640
-4.6521,-0.3716,0.4504,0.0704
-4.5278,-0.3804,0.4521,0.0721
-4.4037,-0.3888,0.4525,0.0671
-4.2796,-0.3960,0.4528,0.0577
-4.1555,-0.4019,0.4529,0.0445
-4.0314,-0.4043,0.4549,0.0302
-3.9076,-0.4055,0.4575,0.0174
-3.7838,-0.4043,0.4605,0.0062
-3.6605,-0.4019,0.4649,-0.0020
-3.5369,-0.4003,0.4665,-0.0061
-3.4135,-0.3958,0.4713,-0.0077
-3.2908,-0.3935,0.4748,-0.0089
-3.1676,-0.3907,0.4774,-0.0095
-3.0447,-0.3876,0.4799,-0.0118
-2.9214,-0.3844,0.4823,-0.0163
-2.7979,-0.3796,0.4853,-0.0219
-2.6751,-0.3736,0.4904,-0.0280
-2.5522,-0.3683,0.4946,-0.0333
-2.4295,-0.3616,0.5002,-0.0377
-2.3068,-0.3570,0.5031,-0.0407
-2.1839,-0.3516,0.5019,-0.0419
-2.0616,-0.3479,0.4959,-0.0416
-1.9390,-0.3466,0.4909,-0.0403
-1.8164,-0.3466,0.4823,-0.0360
-1.6938,-0.3492,0.4781,-0.0318
-1.5712,-0.3526,0.4630,-0.0275
-1.4484,-0.3595,0.4555,-0.0240
-1.3251,-0.3650,0.4288,-0.0264
-1.2024,-0.3736,0.3570,-0.0240
-1.0782,-0.3805,0.2782,-0.0335
-0.9559,-0.3849,0.2950,-0.0247
-0.8311,-0.4001,0.2292,-0.0247
-0.7094,-0.3942,0.2853,-0.0046
-0.5845,-0.4356,0.1861,0.0272
-0.4627,-0.4160,0.2418,0.0497
-0.3388,-0.4718,0.2406,0.0530
-0.2127,-0.4593,0.2768,0.0418
-0.0886,-0.4636,0.1930,0.0151
"""

#!/usr/bin/env python3
import math
import time
import csv
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.clock import Clock, ClockType
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# (Your race line points string or loader goes here)

LIDAR_X = 0.275             # LiDAR ahead of rear axle (sim.yaml lidar_base_link_to_lidar_tf)
BODY_REAR = -0.1249         # collision box spans [-0.125, 0.455] m about the rear axle (centred on wheelbase)
BODY_FRONT = 0.4551
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
            'max_speed': 8.0,
            'tight_speed': 1.8,          # speed where clearance is smallest (cluster, corner)
            'tight_clearance': 0.06,     # clearance (m) treated as "tightest"
            'open_clearance': 0.24,      # clearance (m) above which max_speed is allowed
            'lateral_accel': 8.0,
            'brake_decel': 7.5,
            'accel': 7.5,
            'obstacle_speed': 0.8,
            'speed_lead': 0.0,           # m ahead on the profile to read the target speed
            'min_speed': 0.4,

            'wheelbase': 0.3302,
            'vehicle_width': 0.31,
            'max_steering': 0.4189,

            'latency_comp': 0.12,        # s; steering servo lag compensation
            'ff_gain': 1.0,              # corner-cutting compensation (0 = plain pure pursuit)
            'ff_preview': 0.0,           # where along the look-ahead to sample curvature (0..1)
            'steer_alpha': 0.85,         # steering smoothing (1.0 = none)
            'min_lookahead': 0.60,
            'max_lookahead': 1.60,
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
            'min_gap_depth': 0.8,        # m of free range a gap direction needs
            'emergency_distance': 0.20,  # bumper-to-obstacle distance that forces min speed

            # Stuck recovery
            'stuck_time': 0.8,
            'reverse_time': 0.8,
            'reverse_speed': -0.6,

            # Startup grace period (the out-lap is not timed, so take it gently)
            'startup_hold': 0.5,         # s of steady scan+odom before the car moves
            'outlap_speed': 3.0,         # speed cap until the cluster has been passed once
            'outlap_end_x': 2.8,         # first time x passes this on the bottom straight -> race pace
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
        self.escape_until = 0.0
        self.escape_side = 1.0
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
        self.create_subscription(LaserScan, '/scan', self.on_scan, rclpy.qos.qos_profile_sensor_data)
        # depth 1: always act on the newest pose, never on a backlog of old ones
        self.create_subscription(Odometry, '/ego_racecar/odom', self.on_odom, 1)
        self.odom_count = 0
        self.create_timer(2.0, self.health_check, clock=Clock(clock_type=ClockType.STEADY_TIME))

        self.create_timer(0.1, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'Stage 2 Controller v9.1 (ROS time fixed) ready: {len(self.waypoints)} race-line points, max_speed={self.max_speed}')

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
        win = 6
        v_clear = np.min(np.stack([np.roll(v_clear, -j) for j in range(-win, win + 1)]), axis=0)

        v = np.minimum(np.minimum(v_curve, v_clear), self.max_speed)
        for _ in range(2):
            for i in range(2 * n, -1, -1):
                j = i % n
                v[j] = min(v[j], math.sqrt(v[(j + 1) % n] ** 2 + 2 * self.brake_decel * self.wp_ds[j]))
            for i in range(2 * n):
                j = i % n
                v[(j + 1) % n] = min(v[(j + 1) % n], math.sqrt(v[j] ** 2 + 2 * self.accel * self.wp_ds[j]))
        self.wp_speed = v

    # ------------------------------------------------------------------ I/O
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
        self.scan_time = self.get_clock().now()

    def fresh(self):
        if self.scan is None or self.scan_time is None or self.odom_time is None:
            return False
        now = self.get_clock().now()
        scan_age = (now - self.scan_time).nanoseconds / 1e9
        odom_age = (now - self.odom_time).nanoseconds / 1e9
        return scan_age <= self.sensor_timeout and odom_age <= self.sensor_timeout

    def health_check(self):
        try:
            pubs = self.count_publishers('/drive')
        except Exception:
            pubs = 1
        if pubs > 1:
            self.get_logger().error(
                f'{pubs} nodes are publishing /drive - another controller or teleop is running.')
        rate = self.odom_count / 2.0
        self.odom_count = 0
        if self.odom_time is not None and 0.0 < rate < 150.0:
            self.get_logger().warn(
                f'odom arriving at {rate:.0f} Hz (expected ~250): the simulator is running slow.')

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
        horizon = max(2.0, 1.2 * v + v * v / (2.0 * self.brake_decel) + 0.8)
        sel = valid & (r < horizon + 1.5)
        if not np.any(sel):
            return math.inf
        lx = LIDAR_X + r[sel] * self._cos[sel]
        ly = r[sel] * self._sin[sel]
        c, s = math.cos(yaw), math.sin(yaw)
        px = x + c * lx - s * ly
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

        ahead = lx > BODY_FRONT + 0.05
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
        deep = extended >= self.min_gap_depth
        if np.any(deep):
            score = np.where(deep, score, -np.inf)
        else:
            score = extended
        best = int(np.argmax(score))
        a = float(ang[best])
        d = clamp(float(extended[best]), 0.5, 1.5)
        tx = LIDAR_X + d * math.cos(a)
        ty = d * math.sin(a)
        steer = math.atan(2.0 * self.wheelbase * ty / (tx * tx + ty * ty))
        return clamp(steer, -self.max_steering, self.max_steering), float(extended[best])

    def arc_free_distance(self, steer, r, valid, max_dist=1.5, margin=0.03):
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
        lx = LIDAR_X + r * self._cos
        ly = r * self._sin
        m = valid & (lx > BODY_FRONT - 0.05) & (np.abs(ly) < BODY_HALF_W + 0.03)
        if not np.any(m):
            return math.inf
        return float(np.min(lx[m])) - BODY_FRONT

    # ------------------------------------------------------------------ recovery
    def update_stuck(self, vx, bumper, r, valid, now_sec):
        if self.reverse_until is not None:
            if now_sec < self.reverse_until:
                return True
            self.reverse_until = None
            self.stuck_since = None
            self.last_steering = 0.0
        if self.last_cmd_speed > 0.2 and abs(vx) < 0.05:
            if self.stuck_since is None:
                self.stuck_since = now_sec
            elif now_sec - self.stuck_since > self.stuck_time and bumper < 0.3:
                left = valid & (self._angles > 0.0) & (self._angles < math.radians(90))
                right = valid & (self._angles < 0.0) & (self._angles > math.radians(-90))
                l_min = float(np.min(r[left])) if np.any(left) else 10.0
                r_min = float(np.min(r[right])) if np.any(right) else 10.0
                self.reverse_steer = self.max_steering if l_min < r_min else -self.max_steering
                self.reverse_until = now_sec + self.reverse_time
                self.get_logger().warn('Stuck - reversing')
                return True
        else:
            self.stuck_since = None
        return False

    # ------------------------------------------------------------------ main loop
    def on_odom(self, msg):
        current_time = self.get_clock().now()
        self.odom_time = current_time
        self.odom_count += 1
        
        if not self.fresh():
            self.sensors_ok_since = None
            self.stop()
            return
            
        now_sec = current_time.nanoseconds / 1e9
        if now_sec - self.last_control < self.control_period - 1e-4:
            return
        self.last_control = now_sec

        # Startup grace: wait for steady sensors before moving
        if self.sensors_ok_since is None:
            self.sensors_ok_since = current_time
        startup_elapsed = (current_time - self.sensors_ok_since).nanoseconds / 1e9
        if startup_elapsed < self.startup_hold:
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

        if self.update_stuck(vx, bumper, r, valid, now_sec):
            self.blocked_count = self.clear_count = 0
            self.last_cmd_speed = 0.0
            self.publish(self.reverse_steer, self.reverse_speed)
            return

        lookahead = clamp(self.min_lookahead + self.lookahead_gain * v,
                          self.min_lookahead, self.max_lookahead)
        tau = self.latency_comp
        yaw_rate = v * math.tan(self.last_steering) / self.wheelbase
        px = pos.x + v * tau * math.cos(yaw + 0.5 * yaw_rate * tau)
        py = pos.y + v * tau * math.sin(yaw + 0.5 * yaw_rate * tau)
        pyaw = yaw + yaw_rate * tau
        pp_steer, target_idx, lx, ly, near_idx = self.pure_pursuit(px, py, pyaw, lookahead, v)

        # LiDAR guard on the race line (once per new scan)
        guard_time_sec = self._guard_time.nanoseconds / 1e9 if isinstance(self._guard_time, rclpy.time.Time) else self._guard_time
        if self.scan_time != self._guard_scan_time and (now_sec - guard_time_sec) >= self.guard_period:
            self._guard_scan_time = self.scan_time
            self._guard_time = current_time
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
                self.gap_since = current_time
                self.get_logger().warn('Race line blocked - follow the gap')
            if self.blocked_s < math.inf:
                self.clear_since = None
            elif self.clear_since is None:
                self.clear_since = current_time
                
            clear_sec = (current_time - self.clear_since).nanoseconds / 1e9 if self.clear_since else 0.0
            if (self.in_obstacle_zone and self.clear_since is not None
                    and clear_sec > self.clear_time
                    and self.arc_free_distance(pp_steer, r, valid) >= self.rejoin_free):
                self.in_obstacle_zone = False
                self.rejoining = True
                self.get_logger().info('Race line clear - pure pursuit')

        if self.in_obstacle_zone:
            gap_since_sec = self.gap_since.nanoseconds / 1e9 if isinstance(self.gap_since, rclpy.time.Time) else self.gap_since
            if v > 0.05:
                self.gap_since = current_time
            elif now_sec - gap_since_sec > self.gap_stuck_time:
                left = valid & (self._angles > 0.0) & (self._angles < math.radians(90))
                right = valid & (self._angles < 0.0) & (self._angles > math.radians(-90))
                l_min = float(np.min(r[left])) if np.any(left) else 10.0
                r_min = float(np.min(r[right])) if np.any(right) else 10.0
                self.reverse_steer = self.max_steering if l_min < r_min else -self.max_steering
                self.reverse_until = now_sec + self.reverse_time
                self.gap_since = current_time + rclpy.duration.Duration(seconds=self.reverse_time)
                self.get_logger().warn('Follow the gap stalled - backing out')
                self.publish(self.reverse_steer, self.reverse_speed)
                return

        if not self.in_obstacle_zone:
            steering = pp_steer
            lead = int(round(self.speed_lead / max(1e-3, float(np.mean(self.wp_ds)))))
            target_speed = float(np.min(self.wp_speed[(near_idx + lead + np.arange(0, 4)) % len(self.wp)]))
            if self.rejoining:
                wx, wy = self.wp[near_idx]
                if math.hypot(pos.x - wx, pos.y - wy) < 0.10:
                    self.rejoining = False
                else:
                    target_speed = min(target_speed, self.rejoin_speed)
                    free = self.arc_free_distance(pp_steer, r, valid)
                    target_speed = min(target_speed,
                                       math.sqrt(2.0 * self.brake_decel * max(0.0, free - 0.05)))
            if self.blocked_s < math.inf:
                stop_room = max(0.0, self.blocked_s - self.stop_buffer)
                target_speed = min(target_speed, math.sqrt(2.0 * self.brake_decel * stop_room))
        else:
            steering, gap_dist = self.disparity_extender_steering(r, valid, math.atan2(ly, lx))
            free = self.arc_free_distance(steering, r, valid)
            if free < 0.15 and bumper < 0.35:
                left = valid & (self._angles > 0.2) & (self._angles < 1.6)
                right = valid & (self._angles < -0.2) & (self._angles > -1.6)
                l_free = float(np.median(r[left])) if np.any(left) else 0.0
                r_free = float(np.median(r[right])) if np.any(right) else 0.0
                side = 1.0 if l_free >= r_free else -1.0
                self.escape_side = side
                self.escape_until = now_sec + self.reverse_time + 1.0
                self.reverse_steer = -side * self.max_steering
                self.reverse_until = now_sec + self.reverse_time
                self.gap_since = current_time + rclpy.duration.Duration(seconds=self.reverse_time)
                self.last_cmd_speed = 0.0
                self.publish(self.reverse_steer, self.reverse_speed)
                return
            if now_sec < self.escape_until and free >= 0.15:
                steering = self.escape_side * self.max_steering
            target_speed = self.obstacle_speed * (1.0 - 0.4 * abs(steering) / self.max_steering)
            if gap_dist < 1.0:
                target_speed = min(target_speed, 0.5)
            target_speed = min(target_speed,
                               math.sqrt(2.0 * self.brake_decel * max(0.0, free - 0.05)))
            target_speed = max(target_speed, 0.25) if free > 0.15 else target_speed

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
