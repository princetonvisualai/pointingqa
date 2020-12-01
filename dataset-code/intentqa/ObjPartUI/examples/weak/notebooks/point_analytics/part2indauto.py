import collections


def part2ind():
    pimap = collections.defaultdict(dict)

    # Class: aeroplane

    pimap[2][100] = 100  # silhouette
    pimap[2][120] = 120  # body
    pimap[2][130] = 130  # left_wing
    pimap[2][140] = 130  # right_wing
    pimap[2][151] = 151  # engine_1
    pimap[2][152] = 151  # engine_2
    pimap[2][153] = 151  # engine_3
    pimap[2][154] = 151  # engine_4
    pimap[2][155] = 151  # engine_5
    pimap[2][156] = 151  # engine_6
    pimap[2][161] = 161  # wheel_1
    pimap[2][162] = 161  # wheel_2
    pimap[2][163] = 161  # wheel_3
    pimap[2][164] = 161  # wheel_4
    pimap[2][165] = 161  # wheel_5
    pimap[2][166] = 161  # wheel_6
    pimap[2][167] = 161  # wheel_7
    pimap[2][168] = 161  # wheel_8
    pimap[2][170] = 170  # stern

    # Class: bicycle

    pimap[23][100] = 100  # silhouette
    pimap[23][110] = 110  # chainwheel
    pimap[23][120] = 120  # body
    pimap[23][130] = 130  # front_wheel
    pimap[23][140] = 130  # back_wheel
    pimap[23][151] = 151  # headlight_1
    pimap[23][152] = 151  # headlight_2
    pimap[23][153] = 151  # headlight_3
    pimap[23][160] = 160  # handlebar
    pimap[23][170] = 170  # saddle

    # Class: bird

    pimap[25][100] = 100  # silhouette
    pimap[25][110] = 110  # head
    pimap[25][111] = 111  # left_ear
    pimap[25][112] = 111  # right_ear
    pimap[25][113] = 113  # left_eye
    pimap[25][114] = 113  # right_eye
    pimap[25][115] = 115  # left_horn
    pimap[25][116] = 115  # right_horn
    pimap[25][117] = 117  # muzzle
    pimap[25][118] = 117  # beak
    pimap[25][119] = 117  # nose
    pimap[25][120] = 120  # torso
    pimap[25][121] = 121  # neck
    pimap[25][130] = 130  # left_wing
    pimap[25][140] = 130  # right_wing
    pimap[25][150] = 150  # left_leg
    pimap[25][153] = 153  # left_foot
    pimap[25][160] = 150  # right_leg
    pimap[25][163] = 153  # right_foot
    pimap[25][170] = 170  # tail

    # Class: boat

    pimap[31][100] = 100  # silhouette

    # Class: bottle

    pimap[34][100] = 100  # silhouette
    pimap[34][110] = 110  # cap
    pimap[34][120] = 120  # body

    # Class: bus

    pimap[45][100] = 100  # silhouette
    pimap[45][111] = 111  # front_license_plate
    pimap[45][112] = 111  # back_license_plate
    pimap[45][121] = 121  # back_side
    pimap[45][122] = 121  # front_side
    pimap[45][123] = 123  # left_side
    pimap[45][124] = 123  # right_side
    pimap[45][125] = 125  # roof_side
    pimap[45][126] = 126  # door_1
    pimap[45][127] = 126  # door_2
    pimap[45][128] = 126  # door_3
    pimap[45][129] = 126  # door_4
    pimap[45][131] = 131  # left_mirror
    pimap[45][132] = 131  # right_mirror
    pimap[45][151] = 151  # headlight_1
    pimap[45][152] = 151  # headlight_2
    pimap[45][153] = 151  # headlight_3
    pimap[45][154] = 151  # headlight_4
    pimap[45][155] = 151  # headlight_5
    pimap[45][156] = 151  # headlight_6
    pimap[45][157] = 151  # headlight_7
    pimap[45][158] = 151  # headlight_8
    pimap[45][161] = 161  # wheel_1
    pimap[45][162] = 161  # wheel_2
    pimap[45][163] = 161  # wheel_3
    pimap[45][164] = 161  # wheel_4
    pimap[45][165] = 161  # wheel_5
    pimap[45][170] = 170  # window_10
    pimap[45][171] = 170  # window_1
    pimap[45][172] = 170  # window_2
    pimap[45][173] = 170  # window_3
    pimap[45][174] = 170  # window_4
    pimap[45][175] = 170  # window_5
    pimap[45][176] = 170  # window_6
    pimap[45][177] = 170  # window_7
    pimap[45][178] = 170  # window_8
    pimap[45][179] = 170  # window_9
    pimap[45][180] = 170  # window_20
    pimap[45][181] = 170  # window_11
    pimap[45][182] = 170  # window_12
    pimap[45][183] = 170  # window_13
    pimap[45][184] = 170  # window_14
    pimap[45][185] = 170  # window_15
    pimap[45][186] = 170  # window_16
    pimap[45][187] = 170  # window_17
    pimap[45][188] = 170  # window_18
    pimap[45][189] = 170  # window_19

    # Class: car

    pimap[59][100] = 100  # silhouette
    pimap[59][111] = 111  # front_license_plate
    pimap[59][112] = 111  # back_license_plate
    pimap[59][121] = 121  # back_side
    pimap[59][122] = 121  # front_side
    pimap[59][123] = 123  # left_side
    pimap[59][124] = 123  # right_side
    pimap[59][125] = 125  # roof_side
    pimap[59][126] = 126  # door_1
    pimap[59][127] = 126  # door_2
    pimap[59][128] = 126  # door_3
    pimap[59][129] = 126  # door_4
    pimap[59][131] = 131  # left_mirror
    pimap[59][132] = 131  # right_mirror
    pimap[59][151] = 151  # headlight_1
    pimap[59][152] = 151  # headlight_2
    pimap[59][153] = 151  # headlight_3
    pimap[59][154] = 151  # headlight_4
    pimap[59][155] = 151  # headlight_5
    pimap[59][156] = 151  # headlight_6
    pimap[59][157] = 151  # headlight_7
    pimap[59][158] = 151  # headlight_8
    pimap[59][161] = 161  # wheel_1
    pimap[59][162] = 161  # wheel_2
    pimap[59][163] = 161  # wheel_3
    pimap[59][164] = 161  # wheel_4
    pimap[59][165] = 161  # wheel_5
    pimap[59][170] = 170  # window_10
    pimap[59][171] = 170  # window_1
    pimap[59][172] = 170  # window_2
    pimap[59][173] = 170  # window_3
    pimap[59][174] = 170  # window_4
    pimap[59][175] = 170  # window_5
    pimap[59][176] = 170  # window_6
    pimap[59][177] = 170  # window_7
    pimap[59][178] = 170  # window_8
    pimap[59][179] = 170  # window_9
    pimap[59][180] = 170  # window_20
    pimap[59][181] = 170  # window_11
    pimap[59][182] = 170  # window_12
    pimap[59][183] = 170  # window_13
    pimap[59][184] = 170  # window_14
    pimap[59][185] = 170  # window_15
    pimap[59][186] = 170  # window_16
    pimap[59][187] = 170  # window_17
    pimap[59][188] = 170  # window_18
    pimap[59][189] = 170  # window_19

    # Class: cat

    pimap[65][100] = 100  # silhouette
    pimap[65][110] = 110  # head
    pimap[65][111] = 111  # left_ear
    pimap[65][112] = 111  # right_ear
    pimap[65][113] = 113  # left_eye
    pimap[65][114] = 113  # right_eye
    pimap[65][115] = 115  # left_horn
    pimap[65][116] = 115  # right_horn
    pimap[65][117] = 117  # muzzle
    pimap[65][118] = 117  # beak
    pimap[65][119] = 117  # nose
    pimap[65][120] = 120  # torso
    pimap[65][121] = 121  # neck
    pimap[65][130] = 130  # left_front_leg
    pimap[65][131] = 130  # left_front_lower_leg
    pimap[65][132] = 130  # left_front_upper_leg
    pimap[65][133] = 133  # left_front_paw
    pimap[65][140] = 130  # right_front_leg
    pimap[65][141] = 130  # right_front_lower_leg
    pimap[65][142] = 130  # right_front_upper_leg
    pimap[65][143] = 133  # right_front_paw
    pimap[65][150] = 130  # left_back_leg
    pimap[65][151] = 130  # left_back_lower_leg
    pimap[65][152] = 130  # left_back_upper_leg
    pimap[65][153] = 133  # left_back_paw
    pimap[65][160] = 130  # right_back_leg
    pimap[65][161] = 130  # right_back_lower_leg
    pimap[65][162] = 130  # right_back_upper_leg
    pimap[65][163] = 133  # right_back_paw
    pimap[65][170] = 170  # tail

    # Class: chair

    pimap[72][100] = 100  # silhouette

    # Class: cow

    pimap[98][100] = 100  # silhouette
    pimap[98][110] = 110  # head
    pimap[98][111] = 111  # left_ear
    pimap[98][112] = 111  # right_ear
    pimap[98][113] = 113  # left_eye
    pimap[98][114] = 113  # right_eye
    pimap[98][115] = 115  # left_horn
    pimap[98][116] = 115  # right_horn
    pimap[98][117] = 117  # muzzle
    pimap[98][118] = 117  # beak
    pimap[98][119] = 117  # nose
    pimap[98][120] = 120  # torso
    pimap[98][121] = 121  # neck
    pimap[98][130] = 130  # left_front_leg
    pimap[98][131] = 130  # left_front_lower_leg
    pimap[98][132] = 130  # left_front_upper_leg
    pimap[98][140] = 130  # right_front_leg
    pimap[98][141] = 130  # right_front_lower_leg
    pimap[98][142] = 130  # right_front_upper_leg
    pimap[98][150] = 130  # left_back_leg
    pimap[98][151] = 130  # left_back_lower_leg
    pimap[98][152] = 130  # left_back_upper_leg
    pimap[98][160] = 130  # right_back_leg
    pimap[98][161] = 130  # right_back_lower_leg
    pimap[98][162] = 130  # right_back_upper_leg
    pimap[98][170] = 170  # tail

    # Class: dog

    pimap[113][100] = 100  # silhouette
    pimap[113][110] = 110  # head
    pimap[113][111] = 111  # left_ear
    pimap[113][112] = 111  # right_ear
    pimap[113][114] = 114  # right_eye
    pimap[113][115] = 115  # left_horn
    pimap[113][116] = 115  # right_horn
    pimap[113][117] = 117  # muzzle
    pimap[113][118] = 117  # beak
    pimap[113][119] = 117  # nose
    pimap[113][120] = 120  # torso
    pimap[113][121] = 121  # neck
    pimap[113][130] = 130  # left_front_leg
    pimap[113][131] = 130  # left_front_lower_leg
    pimap[113][132] = 130  # left_front_upper_leg
    pimap[113][133] = 133  # left_front_paw
    pimap[113][140] = 130  # right_front_leg
    pimap[113][141] = 130  # right_front_lower_leg
    pimap[113][142] = 130  # right_front_upper_leg
    pimap[113][143] = 133  # right_front_paw
    pimap[113][150] = 130  # left_back_leg
    pimap[113][151] = 130  # left_back_lower_leg
    pimap[113][152] = 130  # left_back_upper_leg
    pimap[113][153] = 133  # left_back_paw
    pimap[113][160] = 130  # right_back_leg
    pimap[113][161] = 130  # right_back_lower_leg
    pimap[113][162] = 130  # right_back_upper_leg
    pimap[113][163] = 133  # right_back_paw
    pimap[113][170] = 170  # tail
    pimap[113][190] = 114  # left_eye

    # Class: horse

    pimap[207][100] = 100  # silhouette
    pimap[207][110] = 110  # head
    pimap[207][111] = 111  # left_ear
    pimap[207][112] = 111  # right_ear
    pimap[207][113] = 113  # left_eye
    pimap[207][114] = 113  # right_eye
    pimap[207][117] = 117  # muzzle
    pimap[207][118] = 117  # beak
    pimap[207][119] = 117  # nose
    pimap[207][120] = 120  # torso
    pimap[207][121] = 121  # neck
    pimap[207][130] = 130  # left_front_leg
    pimap[207][131] = 130  # left_front_lower_leg
    pimap[207][132] = 130  # left_front_upper_leg
    pimap[207][133] = 133  # left_front_hoof
    pimap[207][140] = 130  # right_front_leg
    pimap[207][141] = 130  # right_front_lower_leg
    pimap[207][142] = 130  # right_front_upper_leg
    pimap[207][143] = 133  # right_front_hoof
    pimap[207][150] = 130  # left_back_leg
    pimap[207][151] = 130  # left_back_lower_leg
    pimap[207][152] = 130  # left_back_upper_leg
    pimap[207][153] = 133  # left_back_hoof
    pimap[207][160] = 130  # right_back_leg
    pimap[207][161] = 130  # right_back_lower_leg
    pimap[207][162] = 130  # right_back_upper_leg
    pimap[207][163] = 133  # right_back_hoof
    pimap[207][170] = 170  # tail

    # Class: motorbike

    pimap[258][100] = 100  # silhouette
    pimap[258][110] = 110  # chainwheel
    pimap[258][120] = 120  # body
    pimap[258][130] = 130  # front_wheel
    pimap[258][140] = 130  # back_wheel
    pimap[258][151] = 151  # headlight_1
    pimap[258][152] = 151  # headlight_2
    pimap[258][153] = 151  # headlight_3
    pimap[258][160] = 160  # handlebar
    pimap[258][170] = 170  # saddle

    # Class: person

    pimap[284][100] = 100  # silhouette
    pimap[284][110] = 110  # head
    pimap[284][111] = 111  # left_ear
    pimap[284][112] = 111  # right_ear
    pimap[284][113] = 113  # left_eye
    pimap[284][114] = 113  # right_eye
    pimap[284][115] = 115  # left_eyebrow
    pimap[284][116] = 115  # right_eyebrow
    pimap[284][117] = 117  # mouth
    pimap[284][118] = 118  # hair
    pimap[284][119] = 119  # nose
    pimap[284][120] = 120  # torso
    pimap[284][121] = 121  # neck
    pimap[284][131] = 131  # left_lower_arm
    pimap[284][132] = 131  # left_upper_arm
    pimap[284][133] = 133  # left_hand
    pimap[284][141] = 131  # right_lower_arm
    pimap[284][142] = 131  # right_upper_arm
    pimap[284][143] = 133  # right_hand
    pimap[284][151] = 151  # left_lower_leg
    pimap[284][152] = 151  # left_upper_leg
    pimap[284][153] = 153  # left_foot
    pimap[284][161] = 151  # right_lower_leg
    pimap[284][162] = 151  # right_upper_leg
    pimap[284][163] = 153  # right_foot

    # Class: pottedplant

    pimap[308][100] = 100  # silhouette
    pimap[308][110] = 110  # plant
    pimap[308][120] = 120  # pot

    # Class: sheep

    pimap[347][100] = 100  # silhouette
    pimap[347][110] = 110  # head
    pimap[347][111] = 111  # left_ear
    pimap[347][112] = 111  # right_ear
    pimap[347][113] = 113  # left_eye
    pimap[347][114] = 113  # right_eye
    pimap[347][115] = 115  # left_horn
    pimap[347][116] = 115  # right_horn
    pimap[347][117] = 117  # muzzle
    pimap[347][118] = 117  # beak
    pimap[347][119] = 117  # nose
    pimap[347][120] = 120  # torso
    pimap[347][121] = 121  # neck
    pimap[347][130] = 130  # left_front_leg
    pimap[347][131] = 130  # left_front_lower_leg
    pimap[347][132] = 130  # left_front_upper_leg
    pimap[347][140] = 130  # right_front_leg
    pimap[347][141] = 130  # right_front_lower_leg
    pimap[347][142] = 130  # right_front_upper_leg
    pimap[347][150] = 130  # left_back_leg
    pimap[347][151] = 130  # left_back_lower_leg
    pimap[347][152] = 130  # left_back_upper_leg
    pimap[347][160] = 130  # right_back_leg
    pimap[347][161] = 130  # right_back_lower_leg
    pimap[347][162] = 130  # right_back_upper_leg
    pimap[347][170] = 170  # tail

    # Class: sofa

    pimap[368][100] = 100  # silhouette

    # Class: dining_table

    pimap[397][100] = 100  # silhouette

    # Class: train

    pimap[416][100] = 100  # silhouette
    pimap[416][110] = 110  # head
    pimap[416][111] = 111  # headlight_1
    pimap[416][112] = 111  # headlight_2
    pimap[416][113] = 111  # headlight_3
    pimap[416][114] = 114  # head_front_side
    pimap[416][115] = 115  # head_left_side
    pimap[416][116] = 115  # head_right_side
    pimap[416][117] = 117  # head_roof_side
    pimap[416][121] = 121  # coach_1
    pimap[416][122] = 121  # coach_2
    pimap[416][123] = 121  # coach_3
    pimap[416][124] = 121  # coach_4
    pimap[416][125] = 121  # coach_5
    pimap[416][126] = 121  # coach_6
    pimap[416][127] = 121  # coach_7
    pimap[416][128] = 121  # coach_8
    pimap[416][129] = 121  # coach_9
    pimap[416][131] = 131  # coach_roof_side_1
    pimap[416][132] = 131  # coach_roof_side_2
    pimap[416][133] = 131  # coach_roof_side_3
    pimap[416][134] = 131  # coach_roof_side_4
    pimap[416][135] = 131  # coach_roof_side_5
    pimap[416][136] = 131  # coach_roof_side_6
    pimap[416][137] = 131  # coach_roof_side_7
    pimap[416][138] = 131  # coach_roof_side_8
    pimap[416][139] = 131  # coach_roof_side_9
    pimap[416][141] = 141  # coach_back_side_1
    pimap[416][142] = 141  # coach_back_side_2
    pimap[416][143] = 141  # coach_back_side_3
    pimap[416][144] = 141  # coach_back_side_4
    pimap[416][145] = 141  # coach_back_side_5
    pimap[416][146] = 141  # coach_back_side_6
    pimap[416][147] = 141  # coach_back_side_7
    pimap[416][148] = 141  # coach_back_side_8
    pimap[416][149] = 141  # coach_back_side_9
    pimap[416][171] = 171  # coach_left_side_1
    pimap[416][172] = 171  # coach_left_side_2
    pimap[416][173] = 171  # coach_left_side_3
    pimap[416][174] = 171  # coach_left_side_4
    pimap[416][175] = 171  # coach_left_side_5
    pimap[416][176] = 171  # coach_left_side_6
    pimap[416][177] = 171  # coach_left_side_7
    pimap[416][178] = 171  # coach_left_side_8
    pimap[416][179] = 171  # coach_left_side_9
    pimap[416][181] = 171  # coach_right_side_1
    pimap[416][182] = 171  # coach_right_side_2
    pimap[416][183] = 171  # coach_right_side_3
    pimap[416][184] = 171  # coach_right_side_4
    pimap[416][185] = 171  # coach_right_side_5
    pimap[416][186] = 171  # coach_right_side_6
    pimap[416][187] = 171  # coach_right_side_7
    pimap[416][188] = 171  # coach_right_side_8
    pimap[416][189] = 171  # coach_right_side_9

    # Class: tvmonitor

    pimap[427][100] = 100  # silhouette
    pimap[427][110] = 110  # screen
    pimap[427][120] = 120  # framescreen
    pimap[427][170] = 170  # frame
    return dict(pimap)
