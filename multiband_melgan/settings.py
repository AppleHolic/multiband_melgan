SAMPLE_RATE: int = 22050  # sample rate of target wave
WIN_LENGTH: int = 1024  # STFT window length
N_FFT: int = 1024
HOP_LENGTH: int = 256  # STFT hop length
HOP_STRIDE: int = WIN_LENGTH // HOP_LENGTH  # frames per window
SPEC_SIZE: int = WIN_LENGTH // 2 + 1  # spectrogram bands
MEL_SIZE: int = 80  # mel-spectrogram bands
MEL_MIN: int = 80  # mel minimum freq.
MEL_MAX: int = 7600  # mel maximum freq.


#
# VCTK Mel Spec Stats
#
VCTK_MEL_MEAN = [-1.7840925522920925, -1.824937665295949, -1.8769061518640275, -1.9285995846521637,
                 -2.0427474028292427, -2.1888718447332067, -2.1912275620326023, -2.1289233943585693,
                 -2.171573636188001, -2.183868609314208, -2.173621955601833, -2.2400926374713315,
                 -2.2690427622079965, -2.3093953097595006, -2.4092646419860335, -2.4534265352578424,
                 -2.482956320478888, -2.5723153474977702, -2.605068131007688, -2.5928338929855372,
                 -2.667521266364133, -2.6686454464627305, -2.662492633379604, -2.707092259682377,
                 -2.6773633295805292, -2.7054625610496803, -2.710006667232406, -2.7129597027974803,
                 -2.723884686978159, -2.7356378334903284, -2.746209658972939, -2.7584107420980812,
                 -2.7719337946016385, -2.783040530805424, -2.777976719044007, -2.800787017449506,
                 -2.773485340195773, -2.795704162801426, -2.795743093215834, -2.8028609644765847,
                 -2.812700683061219, -2.8238131714032177, -2.838795139267604, -2.8583230573693847,
                 -2.866159966921329, -2.9068262736248034, -2.909316226440284, -2.9223899168372305,
                 -2.939863452224247, -2.93715328604993, -2.9552121029784457, -2.9692611509876197,
                 -2.966732165676911, -2.976352332347711, -2.980525432631505, -2.9835633998871622,
                 -2.9875912601582413, -2.9955748924161485, -3.001136676617709, -3.0150156454989374,
                 -3.031560355909347, -3.0637496331098713, -3.0942035062742534, -3.1250574037310446,
                 -3.1595236166192406, -3.197667499062203, -3.2304028920633487, -3.265306762326591,
                 -3.30065691584818, -3.3485079002253535, -3.3862128395330147, -3.4165720540913282,
                 -3.437771949351909, -3.4519111254257244, -3.4590990846868643, -3.4625444132338927,
                 -3.466759295475555, -3.4689337188702054, -3.4691459637102136, -3.477886229999893]


VCTK_MEL_STD = [0.6861617158355379, 0.8894829839738728, 1.094622033308182, 1.1242213544506143,
                1.048701035717745, 0.9834663138472942, 1.021459266073759, 1.0882311430052352,
                1.1111999210242818, 1.1226369697248089, 1.124836999648175, 1.1192962815793583,
                1.1117271861143605, 1.0894580368641853, 1.066361738686444, 1.0529229487868752,
                1.0103930912189647, 0.9931615403185517, 0.9847407711673594, 0.9174889488781374,
                0.935118462507135, 0.9522947336783119, 0.9220502034481447, 0.94942401469872,
                0.9641408914337891, 0.9736663808322129, 0.9839325614642865, 0.9845338914608218,
                0.9854540683859567, 0.9914949687858687, 0.9939184320845269, 0.9958759403129132,
                0.9947883109552844, 0.9933869125140862, 0.9955493085220788, 0.9993646165443998,
                1.0066248901794566, 1.0080987745836825, 1.0102315223574008, 1.0088190069751035,
                1.008602278013548, 1.0066101562003869, 1.0014535704477314, 0.9980002292406024,
                0.9921275956747925, 0.9828354126558683, 0.9818242263550642, 0.9807977524810778,
                0.9809087993421169, 0.9907016568570686, 0.9925619527833783, 0.9924795464089928,
                0.9964854281038046, 0.9934017975992241, 0.9938287109119572, 0.9921456098773118,
                0.9893912052215408, 0.9861127687060783, 0.9821178470938837, 0.9754930398816252,
                0.9699329533317232, 0.9580052755387448, 0.9464439963590533, 0.935846839910302,
                0.91934909798266, 0.9012991425365203, 0.883826699323533, 0.8649905928124731,
                0.8452182176123265, 0.8286812365859492, 0.8163144897677547, 0.8094039507968079,
                0.8029856276167318, 0.800662551268947, 0.8009980783154776, 0.8015511399645605,
                0.7977596875572289, 0.7920179000628864, 0.7887358027181898, 0.7795109608362336]
