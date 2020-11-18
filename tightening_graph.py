
from datetime import datetime
import pyodbc
from dateutil.parser import parse
import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import asarray, int32, float32, frombuffer, ndarray
import pandas as pd
from matplotlib import pyplot
import struct


# abnormal
hex = '0x00E43B40006E70400055A5400085CF40002AF3408083174180E62C4180405041803F654100BB784140DC834100F992414073A14100CEAE4100EFB5414000C9410060D141C079D4410087D741C0F8DC4180A3CC41C0E4A54100096C4100E611410020994000E21A400018AB3F0018763F00C4863F0030A73F0024903F00E0123F0020643E00A0253E0030C03E0020FD3E00D09D3E00E05D3E00207D3E0070943E00F0873E00606A3E00E0763E00F0873E00D0843E00207D3E00308E3E0070AD3E00A0F03E00100B3F00C00F3F0050113F00601F3F00103D3F00185D3F00E8643F00C8613F005C8F3F004C014000682640009616400014024000F2284000BFA44080E8024180372C4180B04E4180456441803C724100B57941804B7C41804B7C4180E77B4180647C41808A7E4100338141C09D8341002E86410054884140EA894180EA8A4180358B41005B8B41804E8B4100F78A41C08C8A41404E8A4140358A4140678A4100DE8A4100A68B4180488C4100B98C4180298D4180748D4140EB8D41C05B8E4140B38E4140178F41C0878F4140F88F418062904100BA9041802A914100CD9141007C9241C0889341C0B494414006964180839741008A98414071994140209A4140CF9A41C08A9B4100409C4180E29C4100B79D4100989E41407F9F414047A041C01BA1410003A24100CBA2418086A3414061A441C080A54140D2A6410043A84100BAA941C043AB4140E0AC410083AE414032B0414090B14180C2B24100E2B34100AAB441C084B541C065B641C02DB741002EB841000FB94100F0B94100D1BA41409FBB41C05ABC41402FBD418016BE41001DBF4100FEBF41C0F1C0414011C241403DC3418088C44180E6C541C04AC741C0C1C8418032CA41009DCB41C0DBCC41C0D5CD4100EFCE4100D0CF4100E3D04140CAD141400FD341C02ED4418022D54100BFD641C0CBD74140B9D841C0BFD94180CCDA4100D3DB4100B4DC41C0F2DD4140E0DE41C069DF41001FE0410019E14180D4E141407DE241C01FE341C0CEE341C064E4414007E54100C9E5410078E641801AE74100D6E741408BE841403AE941401BEA4140E3EA418098EB41802EEC4180ABEC410035ED4100CBED41C08CEE414016EF4140C5EF41C067F0418029F14100FEF14180EBF24140ADF341C04FF44140C0F441C049F541C0F8F54100AEF6418082F7418018F84180AEF841C031F941C0C7F941406AFA41C025FB4140AFFB418064FC410007FD4180A9FD41007EFE41C08AFF41E0350042E099004280F4004240520142C0A9014280070242A06E024220DF02426062034240C303420021044200850442C0FB0442A08E054220180642407F064240FC06424060074200BE0742A0310842E09B084220060942C079094200E40942806D0A42A0ED0A4220770B4220F40B4240420C4200A00C4200040D42805B0D4220B60D42E02C0E4260840E4260E80E42C06E0F4240DF0F42C04F104220D61042A02D1142E07E114280C01142800B1242004A1242C08E1242C0F21242204713422092134260CA1342E00814426060144240A81442C01815424070154200E7154220671642A0D71642A03B1742809C17428019184200A31842E035194220B91942C05E1A4200FB1A4280841B42A0041C42A0811C4220F21C4240591D4280AA1D4280F51D42C0461E42A0A71E4260051F4260501F4220AE1F42C021204200A5204280CA204200531D42C0E6144260A80842C0BBFB41C01FE3414083C841808FAE4180599E414079A041C0B5B141802CCB414058E4418031F841C02602422082054260A80842A0380B42808D0D42C0870F4260271142C0DF114220851142A0B01042C0D20F42C03C0F42801D0F42005C0F42C06E0F42407B0F4280810F4280680F4200110F42A08A0E4260200E4260D50D42805B0D42C0670C42604B0B42C0410A428073094220D40842602B0842803407426037064220500542009E044240F5034200AA024220B900428090FD41C0F9F941C0C0F641C068F4410017F241C035F041C03BEF41404EEE410003ED41C008EB414021E94180FBE741807EE74100BDE741C01AE841C0E8E7418097E7418001E7414084E5418038E241C024DE41C042D941803BD54100DDD14100E9CF41C020CE41409DCC4100B6CB41807DCA4140CEC8418018C641C036C24180DEBE41001CBB41809EB84140BDB6418001B541007EB34140F4B1414032B041C082AD4180ADA941C016A641808CA241407F9F4100219D41C0DB9A4100BC984100C8964140DA944100AE9241C036904180748D41C08C8A418047884180E9864180538641007F8541C07E8441C06B8341407E8241004C814180BC7E41001F794180817341801C6F41003B6C4100E36941809D6641003F634100ED5F4180245D4180685A4100A65641809E514100014C4180FF4541007B404180283B4100A435418051304180EC2B410043284180B22441804D204100271C418000184100441341806E0E4100B8084180CF024100FAFA4000FEF14000E3E9400026E3400088DB400003D4400078CD400000C84000BAC24000FDBB400014B44000F9AB40008DA440003A9D400083954000B38D4000928640001A8140003E784000166E400058634000CC584000084F4000AE444000223A4000442C400096164000E8FD3F0084CB3F00C8963F00D8563F0060063F00308E3E00C0DA3D00803B3D00007A3C0000163C0000AF3C00007A3C00007A3C0000AF3C0000E13C0000E13C0000E13C0080093D0000E13C0080093D0080093D0000E13C0000E13C0000AF3C0000163C0000163C0000163C00000000000048BC000096BC0000FABC0000FABC0000FABC0000FABC000016BD000048BD000048BD000048BD000048BD000048BD000048BD000048BD000048BD000048BD000048BD000016BD0000FABC0000FABC000096BC000048BC0000C8BC0000C8BC000048BC000048BC000048BC000000000000163C0000163C0000163C0000163C0000163C0000163C0000163C0000163C0000163C0000163C0000163C0000163C000048BC000048BB000048BB000048BC000048BC000048BC000048BC000048BC000048BC000048BC000048BC000048BC000048BC000096BC0000FABC0000FABC0000C8BC000048BC000096BC0000C8BC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC000048BC000048BC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC0000FABC000096BC000048BC000048BC000096BC000048BC000096BC000048BC000096BC000096BC000048BC000048BC000048BC000048BC000048BC000048BC000048BC000048BC'
hex2 = '0x00000000541CD9407EAA2F4126A36441CE3B9341C11AAD41EB18CD41614DE641AC8B0242A8D00E4271C91D4204C32942AC73384282354442329452429F0A5E42C9106C4254047642C9087E42BD5780422A168042A8887E42B98C7B42DAB4784286A875422EF17342329C724219027242E7957142B9647142613D7142FC2C7142FC2C7142193A7142C54D71422A5E71428F6E7142F37E7142588F7142769C7142DAAC7142F8B971425CCA71427AD77142DEE77142FCF4714261057242370F72429B1F72422A267242B92C724200307242473372424733724200307242B92C724247337242DEAF72420836744232BC754276DC774208AE79425CF27B42D2B67D429FA27F42E3768042BD1F8142A888814293F18142C93482425C7682428AA7824247DF82422E0D83428F4683427674834232AC8342D2D68342A4098442112C84420D558442327484424398844222B4844232D8844211F4844222188542A43585429F5E8542C57D854265A885422EC9854286F08542080E864276308642544C86421D6D8642B48586427EA68642B9C08642DEDF8642BDFB8642861C8742C13687422E59874269738742D695874258B3874269D7874247F38742C9108842612988423F4588428F5A8842C9748842198A884254A48842A4B9884297D088429FE2884237FB88429B0B8942471F8942082E894211408942D24E8942DA6089429B6F894200808942C18E8942269F8942E7AD89424CBE894269CB894286D88942B9E089428FEA89421DF18942ACF789423BFE8942C9048A42B4098A4243108A422E158A42191A8A42611D8A424C228A4293258A42DA288A42222C8A42692F8A42B0328A429B378A42E33A8A4271418A42B9448A42A4498A4232508A421D558A42085A8A42505D8A42505D8A4297608A4297608A423B628A423B628A42DE638A4282658A4226678A42C9688A426D6A8A426D6A8A42116C8A42116C8A42B46D8A42B46D8A42FC708A42FC708A42FC708A429F728A42E7758A428A778A42D27A8A42BD7F8A42A8848A4293898A4222908A42B0968A4286A08A42B9A88A4232B48A42ACBF8A426DCE8A422EDD8A42DAF08A423F018B428F168B42F3268B42433C8B424C4E8B423F658B42A4758B42978C8B4243A08B427EBA8B422ACE8B42ACEB8B429F028C42DA1C8C422A328C42C14A8C426D5E8C42A8788C423F918C42C1AE8C42B4C58C424CDE8C423FF58C42C1128D42582B8D427E4A8D4215638D4297808D42D29A8D423FBD8D4265DC8D4204078E42B92C8E429F5A8E42F8818E42DEAF8E4293D58E42C1068F42BD2F8F428F628F428A8B8F4200C08F42FCE88F42711D9042584B9042B9849042FCB0904215E79042B41191422A469142266F9142F8A1914250C991427EFA914232209242A8549242477F9242A8B8924232E892424C1E934247479342D27693422A9E9342B4CD9342B0F69342262B9442C5559442828D944269BB94426DF694429B279542E765954215979542A8D89542C10E964254509642268396425CC6964219FE96425041974222749742C9B0974254E09742E7219842A45998427E9E984282D99842EB249942376399428AB3994208FA9942EB509A42C5959A4261E99A42DE2F9B42C1869B423FCD9B42DA209C4211649C4208B69C429BF79C42DA4C9D42B4919D4297E89D42712D9E42B0829E429FC29E42080E9F42F84D9F427EA69F42E7F19F421D35A042B924A0428FCA9F423F519F423F899E42E3C29D421DB19C42CED39B42FC3C9B4226339B4271719B424CB69B42C9FC9B42931D9C420D299C42C5259C42931D9C4204179C4276109C42E7099C4258039C42DEF79B428FE29B42E3CE9B424CB69B429FA29B42508D9B4232809B422A6E9B420D619B42614D9B429F3E9B4265249B42CE0B9B4204EB9A426DD29A42B9AC9A42A8889A4232549A4293299A427AF39942DAC899421D91994293619942D629994293FD9842D6C59842EF979842155398426D16984204CB9742A49197426D4E9742F819974265D896427EAA9642C1729642DA4496423208964219D295422A929542E7659542B9349542A81095421DE1944269BB944297889442F85D94429724944293E9934271A19342116893423723934265F0924204B792421D8992424C569242502D924222FC914282D19142DA9491421D5D91429F1691422AE2904226A79042E37A9042C9449042CE1B904258E78F4271B98F42267B8F4269438F42EBFC8E4276C88E4286888E429F5A8E429B1F8E42FCF48D4286C08D42E7958D4215638D422E358D4286F88C429FCA8C4286948C42D26E8C427A478C42B0268C42B4FD8B42EBDC8B424CB28B42DE8F8B42C5598B42DE2B8B4222F48A4226CB8A42E39E8A42D27A8A42EB4C8A42DA288A42F3FA89422ADA8942E7AD894232888942615589421D29894261F18842C1C68842048F8842C16288424C2E88429708884254DC87429FB6874271858742195E8742EB2C8742930587427ACF8642DAA4864265708642694786428219864271F58542E7C5854232A085424C728542DE4F854286288542BD0785421DDD84420DB98442C98C8442B96884428A3784423210844204DF8342F3BA8342B08E8342E76D8342EB4483427E2283423BF68242CED382422EA98242658882424C528242321C824297C8814276808142761C81427ECA804237638042F80D80428A437F42C5957E42EFC37D4200207D42E34A7C4265A07B4247CB7A4258277A423B52794204AB78422ED977423F357742F8697642DECF7542260B7542C56D74420DA97342F30E734258577242CEC371425C02714243687042A8B06F42AC236F42116C6E423FD56D42A41D6D42618D6C4254DC6B42114C6B4293A16A42DE176A42616D6942F3E66842BD3F6842DEBF67427E2267422EA966425C126642C595654265F86442CE7B64428AEB6342C978634286E862427E7262423BE26142326C6142EFDB6042766C6042C1E25F4247735F4293E95E42617D5E423BFA5D42088E5D422A0E5D4286A85C42372F5C4269D35B42615D5B42DA045B4219925A424C365A42D2C659424C6E5942D2FE5842DAAC58427E4A584286F857422A965742324457428FDE5642DE8F5642822D564219E255424C8655422A3E5542EBE8544211A45442195254428610544247BB53426D765342042B5342B9EC524297A45242DA6C524200285242D2F651423FB5514211845142C5455142261B514269E35042C9B8504254845042B45950428628504276045042D6D94F420DB94F42B4914F4232744F4269534F422E394F42AC1B4F42B9044F427EEA4E42D2D64E4226C34E4208B64E425CA24E423F954E4222884E424C7E4E4276744E429F6A4E4211644E42825D4E42F3564E4265504E421D4D4E421D4D4E421D4D4E421D4D4E4265504E42'
'''
hex = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\x00\x01\xc4\x01;\xfe\x8f\x12F?&\xf2U?\xe3M\x88?d\xb8;?\x12\xb1\x93?v\xc8\x82?\xd9OC?\xc0a\x94?cpd?\xad\xada?(\xde\x98?\xa4\x9b_?\xaa4\x8f?\x9du\xa0?M\x9fs?\x8b\xad\xb0?m\x12\x95?\xaa4\x8f?_\xaf\xba?\xfd\x83\x8e?\xc8\xcf\xaa?d\xb8\xbb?z-\x98?\x19\xd7\xd2?\xb2\xfe\xb9?4U\xb0?MC\xdf?l\xca\xbd?t8\xd4?@\x84\xf0?\xee|\xc8?\xec1\x00@\x9b\xe5\xf1?\xe5\xc6\xda?K\xca\x0c@M\x9f\xf3?K\x9c\x02@B\xe6\x14@\x1bg\x01@\xc4j\x15@,\xe7\x19@\xd6`\x0f@\x8f\xb61@\x1f\xfa @,\x15$@\xbf\x19=@\x98\x9a)@\x86\xd29@\xa06J@=\x95<@\x9b\x89]@\x81SW@\x15\xceQ@\x85\xb8l@/2b@\xff*k@\xdf-\x82@\x9f\xeer@>\x98\x84@\x88\x03\x8c@\\\xa9\x81@\xcde\x92@5\xe2\x96@O\xea\x92@(\x0c\xa3@\xe7\xb2\x9d@\x12\xf6\xa2@\xb6\xd9\xb0@\x9c\xa3\xaa@\xa1\xda\xb5@l\xe1\xc2@8\x8c\xbb@\xbaU\xcb@\xa9_\xd1@}\x1c\xcc@\xfb\xf3\xdf@\x10!\xe5@\x15*\xe6@\xcf:\xf9@\x9f\x05\xf8@\x06Q\x01A\x15|\x07A\xbc*\tA$\xa7\rA\xd8p\x12A\xd5w\x14AK\xf8\x16A\x87H\x1bA\xd5\xa5\x1eA\x1b\xf1\x1fA\xc8\xa1 A\x0e\xed!A\xbf\xa6#A\xb5\x9b#A\xb0\xa9\'AY:"A\xac[\x17A\x1b\xac\x10Ap\xc6\x03A3;\xe3@8\xa3\xc0@\xde\xb7\xa0@e\xbb\x83@3\rY@\'h7@\x7f6\x19@\xcd \x03@\xa0\x92\xde?~\xee\xc1?qw\xaa?\x17\xba\x94?-\x8b\x85?\x00\x00\x00\x00\xba\x91kt\xc4\x01\x00\x00'
hex2 = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\x00\x01\xc4\x01;\xfeDp\xc2>\x19\x15\xf4?\xd5\xcf\\@^\x88\x9f@\xa0f\xd0@\xa2\xe4\x00A\xc3S\x19A\xfd\xe31AO\x95JA\x88%cA\xc2\xb5{A\x8a3\x8aA\xa7{\x96AP\xd4\xa2Am\x1c\xafA\x16u\xbbAK\xde\xc7A\xdc\x15\xd4A\x85n\xe0A.\xc7\xecAJ\x0f\xf9A\xfa\xb3\x02B\xc2\xcf\x08B\xd0\xf3\x0eB% \x15B3D\x1bBBh!BP\x8c\'B_\xb0-Bm\xd43B{\xf89B\x8a\x1c@B\x98@FB`\\LBo\x80RB}\xa4XBE\xc0^B\x0e\xdcdB\x1c\x00kB\x9e\x13qB\xad7wB/K}B{\xb3\x81B`\xc1\x84B!\xcb\x87B\xe2\xd4\x8aB\xc6\xe2\x8dBd\xe8\x90B%\xf2\x93B\xe6\xfb\x96B\x84\x01\x9aBE\x0b\x9dB\xc0\x0c\xa0B^\x12\xa3B\x1f\x1c\xa6B\x9a\x1d\xa9B8#\xacB\xb3$\xafB.&\xb2B\xa9\'\xb5B$)\xb8B{&\xbbB\x19,\xbeBN%\xc1B\xa6"\xc4B\xfe\x1f\xc7B3\x19\xcaBg\x12\xcdB\xbf\x0f\xd0B\xd1\x04\xd3B)\x02\xd6B;\xf7\xd8B)\xe8\xdbB;\xdd\xdeB*\xce\xe1B\x18\xbf\xe4B\x07\xb0\xe7B\xaf\x98\xeaB{\x85\xedBFr\xf0B\x86N\xf3B\xac\t\xf6B\xbb\xa3\xf8B\x01\x08\xfbB\xc5>\xfdB\x08H\xffB\xe5\x91\x00C\xf3f\x01C@%\x02C\xde\xce\x02C\xbba\x03C\xe8\xdf\x03CxK\x04C\xb0\xac\x04C9\xf9\x04Cc\x1c\x05Cu\x1e\x05C\x1d\x14\x05Cn\xff\x04Cx\xe2\x04C`\xc1\x04C$\x9c\x04C\xfax\x04C\xd0U\x04C\xc96\x04C\xe5\x1b\x04C\x12\x03\x04Cu\xf0\x03C\xfa\xe1\x03C\xa2\xd7\x03C\\\xcf\x03C9\xcb\x03C9\xcb\x03CZ\xa7\xb2\x1f\xc4\x01\x00\x00'


hex2 = b'\x00\x00\x00\x00F\x93\xdd@\xce\x82%A)\x07\\A(?\x89A\xe3\xe2\xa3Ajy\xbeA\x1c\x99\xd8A\x00\xa5\xf2A\x07\x1d\x06B\xc0\xd3\x12B\xdcb\x1fB\xc2\xc3+B\xf3\x068B\xa8\'CBS\xfbJB\xdeuRB\xab\xa6VBo\x8aWB$\xa8WB/\xd8VBV\xcbTBANRBC\xa6OBt\xc9LB\xf2\xefIB\x8c-GB\xae\xdeDBt\x1aCB\xda\xbfABa\xbe@Bn\x0f@B\xe6\x9b?B.]?B\xc6B?B,<?B\x13F?B\xe1Y?BIt?B\xff\x91?B\x01\xb3?B\x04\xd4?B\x06\xf5?BU\x19@B\x0b7@B\xc0T@Bur@B\xde\x8c@BF\xa7@B\xae\xc1@B\x17\xdc@B2\xf3@BM\nAB\x9d.ABkBAB9VAB\xd6}AB\x07jAB\xa4\x91AB\xbf\xa8AB\x8d\xbcAB\xa9\xd3AB\x11\xeeAB\xdf\x01BB`\x12BB\xe2"BBc3BB\x97@BB\xcbMBB\xffZBB3hBB\x1brBBO\x7fBB\x83\x8cBB\xd2\xb0BB\xdeUCB\x89CDB\x05fEB\xb4\x95FB\xcc\xdfGB\x97&IB+?JB\xf2CKB\x81\x1aLB\x8d\xbfLB\xe4FMBl\xbaMB\xa5\tNB]HNB\xe0yNB0\x9eNB\xe5\xbbNB\x9b\xd9NB\x19\xc9NB\xb6\xf0NB\xb6\xf0NB\x03\xf4NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\xb6\xf0NBi\xedNB\x1c\xeaNB\xcf\xe6NB\x19\xc9NB\xb3\xcfNBM\xd6NB5\xe0NB\x82\xe3NB\x19\xc9NB2\xbfNBK\xb5NB\x17\xa8NB0\x9eNB\xaf\x8dNB-}NB_iNB\xdeXNB]HNB\x8f4NB\xc0 NB\xf2\x0cNBq\xfcMBV\xe5MB\x87\xd1MB\x06\xc1MB\xeb\xa9MB\x1d\x96MB\xb4{MB\xff]MB\x97CMB\xe1%MB\xdf\x04MBw\xeaLB\xc1\xccLBY\xb2LB\xf1\x97LB\x88}LB\xd3_LB\x84;LB4\x17LBK\xecKB\x14\xbeKB\x90\x8cKBsTKBU\x1cKB\xea\xe0JB\x1a\xacJBIwJB+?JB[\nJB\x8a\xd5IB\xba\xa0IB\x83rIBLDIB\xc9\x12IB\x92\xe4HB\x0f\xb3HB\xd8\x84HB\x08PHB\xb6\nHB\xfe\xcbGB\x12\x80GB&4GBS\xdeFB\xcd\x8bFBG9FB\xc1\xe6EB\xd5\x9aEB\x83UEB\x98\tEBF\xc4DB\xf4~DB\xa29DB\xea\xfaCB\xe5\xb8CB\xe1vCB\x8f1CBV\xe2BB6\x89BB\x160BB\\\xd0AB<wAB\x1c\x1eAB\x14\xbb@B(o@B\xef\x1f@B\x04\xd4?B\xb2\x8e?B\xadL?B[\x07?B"\xb8>B\xe9h>B\xc9\x0f>B\x90\xc0=B\xbdj=B\xea\x14=B\x17\xbf<B\xf7e<B$\x10<BQ\xba;B1a;B\x92\x18;B@\xd3:B\xee\x8d:B\xeaK:B2\r:B\xe0\xc79B\xf4{9B\x0809B\x1c\xe48B~\x9b8B\xdfR8B\xda\x108B"\xd27B\xb7\x967BM[7B|&7B\xab\xf16B(\xc06B\xa4\x8e6B\xbbc6B726B\xb4\x006B\xe3\xcb5B\x12\x975BBb5B\xbe05B:\xff4B\xb7\xcd4B\xff\x8e4B|]4BE/4B\xc1\xfd3B\xd8\xd23B;\xab3B\xec\x863B\x9db3BM>3B\xb1\x163B\x14\xef2B\xc5\xca2B\xc2\xa92BZ\x8f2B?x2B#a2B\x08J2B\xed22B\x85\x182B\xcf\xfa1B\x1a\xdd1Be\xbf1B\x15\x9b1B\xc6v1BvR1B\xda*1B\xf0\xff0BT\xd80Bj\xad0B\x81\x820B\x97W0Ba)0B\xdd\xf7/B\x0c\xc3/B<\x8e/B\xb8\\/B\xcf1/B\xe5\x06/B\xfc\xdb.B_\xb4.Bv\x89.B\x8c^.B\xa33.B\xb9\x08.B\x83\xda-B1\x95-B\xadc-B\xf5$-B>\xe6,B9\xa4,B4b,B\xc9&,B\x12\xe8+B\xa7\xac+B\x89t+B\x06C+B\xcf\x14+B3\xed*BI\xc2*B`\x97*B)i*B@>*B\xf0\x19*B\xee\xf8)B\x85\xde)Bj\xc7)B\x9c\xb3)B\x81\x9c)Be\x85)B\xb0g)B\xaeF)B\xab%)B\\\x01)Br\xd6(B\x89\xab(B\x9f\x80(BiR(B\x7f\'(B\x96\xfc\'B\xac\xd1\'B]\xad\'B&\x7f\'B=T\'B\xed/\'B\x9e\x0b\'B\x83\xf4&B\x99\xc9&B\x8d$&B/:%BK\xfd#B/q"B\x0b\x82 B\x97n\x1eBS&\x1cBA\xca\x19B\xb0~\x17Bl6\x15B_\x1c\x13Bm\x19\x11BKK\x0fB\x16\xc9\rB2\x8c\x0cBl\x87\x0bB,\xd5\nB\x8d\x8c\nBy\xd8\nB\xb9\x8a\x0bB\x95d\x0cB%;\rB~\xe3\rB\xb9S\x0eB\tx\x0eBSZ\x0eB\xb4\x11\x0eB\xad\xae\rB\x8b4\rB\x00\xa0\x0cB\\\x15\x0cB\xd2\x80\x0bB`\xe2\nB\x88J\nB\xe2\x9e\tB\xd6\xf9\x08Bd[\x08B\x0b\xb3\x07Be\x07\x07B\xbf[\x06B\xb3\xb6\x05B\xf4\x14\x05B\x1a\\\x04BX\x99\x03B1\xdd\x02B\xf1*\x02BK\x7f\x01B?\xda\x00B\xcd;\x00B\x1fU\xffAm\x04\xfeA%\xef\xfcAx\xe0\xfbA\xcd\xf2\xfaA\x8b\x1f\xfaA\xe2R\xf9A\xd4\x8c\xf8A)\x9f\xf7A~\xb1\xf6A9\xbd\xf5A\xf5\xc8\xf4A~\xe8\xf3A\x07\x08\xf3A(\r\xf2A\x13\xe4\xf0Ae\xd5\xefA\x1e\xc0\xeeA\xd7\xaa\xedA\x92\xb6\xecA\xe7\xc8\xebAp\xe8\xeaA.\x15\xeaA\xb74\xe9A\x0fh\xe8A\xcc\x94\xe7A$\xc8\xe6A\x15\x02\xe6Am5\xe5A-\x83\xe4A\xea\xaf\xe3A\x0e\xd6\xe2A3\x1d\xe2A%W\xe1A\xe5\xa4\xe0A\xd9\xff\xdfA3T\xdfA\xc1\xb5\xdeA\xb5\x10\xdeA\x0fe\xddA\x9d\xc6\xdcA+(\xdcA\x88\x9d\xdbA~\x19\xdbA\xda\x8e\xdaAk\x11\xdaAa\x8d\xd9A\xf2\x0f\xd9A\x82\x92\xd8A\xdf\x07\xd8A\t\x91\xd7Ah\'\xd7A-\xb7\xd6A\x8bM\xd6A\x1c\xd0\xd5AFY\xd5A=\xd5\xd4A3Q\xd4A\xf8\xe0\xd3AWw\xd3A\x84!\xd3A\xe5\xd8\xd2A\x12\x83\xd2A?-\xd2A:\xeb\xd1A\x9b\xa2\xd1A\xcam\xd1A\xc8L\xd1A,%\xd1A]\x11\xd1A[\xf0\xd0A\xbe\xc8\xd0A\xbc\xa7\xd0A\xba\x86\xd0AQl\xd0A\xe9Q\xd0A\xb5D\xd0A\x817\xd0AM*\xd0A\xe4\x0f\xd0A\xe2\xee\xcfA\xae\xe1\xcfA\xdf\xcd\xcfA\xab\xc0\xcfA\xab\xc0\xcfA\xab\xc0\xcfA\xab\xc0\xcfAw\xb3\xcfA\xa9\x9f\xcfA\x0f\x99\xcfAu\x92\xcfA\xdb\x8b\xcfAu\x92\xcfA\xdb\x8b\xcfA\x0cx\xcfA\xd8j\xcfA\xa4]\xcfA\nW\xcfA\nW\xcfA\xa4]\xcfA\xa4]\xcfA\nW\xcfA\xd6I\xcfApP\xcfA\nW\xcfA\nW\xcfA\xa4]\xcfA\xa4]\xcfA>d\xcfA>d\xcfA\xa4]\xcfA>d\xcfA\xd8j\xcfArq\xcfA\x0cx\xcfA\xdb\x8b\xcfA\x0f\x99\xcfA\xdd\xac\xcfA\xab\xc0\xcfAE\xc7\xcfAz\xd4\xcfA\x14\xdb\xcfA\xe2\xee\xcfA\xb0\x02\xd0A\x18\x1d\xd0A\x1b>\xd0A\xb7e\xd0AT\x8d\xd0A"\xa1\xd0A\x8a\xbb\xd0A\xbe\xc8\xd0A\x8d\xdc\xd0A\xc1\xe9\xd0A\x8f\xfd\xd0A)\x04\xd1A\xc3\n\xd1A\xc3\n\xd1A]\x11\xd1A\x91\x1e\xd1A\xc6+\xd1A.F\xd1A0g\xd1A3\x88\xd1A\x01\x9c\xd1A5\xa9\xd1A\x03\xbd\xd1A8\xca\xd1A\x06\xde\xd1A\xd4\xf1\xd1A\xa2\x05\xd2A\xa0\xe4\xd1Adt\xd1A$\xc2\xd0Az\xd4\xcfA\x98\xb8\xceAN\x82\xcdA\xd1>\xccAP\xda\xcaA\x9e\x89\xc9A\xec8\xc8Al\xd4\xc6A\x88\x97\xc5A\xa4Z\xc4A\xf2\t\xc3Au\xc6\xc1A]|\xc0AGS\xbfA1*\xbeA\x1c\x01\xbdA6\xa3\xbbA\xb8_\xbaAn)\xb9AY\x00\xb8A\x0f\xca\xb6A+\x8d\xb5A'
hex = b'\x99\xfd\xb0@\xcc\xc8\xc3@\x99\xa5\xda@3\xd3\xea@39\tAf\xfa$A\x99\x95=A\x00 UA\xcc,fA\xcc\x1e\x7fAf\x08\x8cA3S\x9dAf\xa2\xafA3\x1c\xbfA\xcc\x96\xd1A\xcce\xdbA\x00u\xf5A\xe6\xf3\x00Bf\xdc\x01B\x80j\x04B\xff\xbf\xffA\x99\x03\xd7A\x00u\x98A\x99g;A\xcc\xac\xb7@\x99Y\xb5?\xcc\xcc\xb3\xbd\xcc\xcc\x94\xbc33\x82=\x99\x19\xdc>\xcc\x0c,?f\xa6\x9f>\xcc\xcc\xf1=ff\xc6=\x00\x00\x9b=\x00\x00:=33\xdf<33!=\x99\x99-=\xcc\xccR=\xcc\xcc\x94=\xcc\xcc\xb3=\xcc\xcc\xd2=\xcc\xcc\xf1=\x99\x99\xeb=ff\xe5=\x00\x00\xd9=33\xdf=\xcc\xcc\xf1=\xcc\xcc\xf1=33\xfe=\x00\x80\x0b>\xcdL\x05>\xcc\xcc\xf1=\x99\x99\xeb=\x00\x00\xd9=\xcc\xcc\xd2=\x99\x99\xcc=\x00\x00\xd9=33\xdf=33\xfe=\xccL$>\x99\x19=>\x00\x80I>3\xb3O>\x00\x00Y>\x00\x80I>3\xb30>f\xe6\x17>33\xfe=\xcc\xcc\xd2=ff\xe5=f\xe6\x17>ffF>\x99\x99k>\xccL\xe2>\xcc,G?3\xf3\x89?f&\xaf?3\xb3\x11@fF\x8c@\x99\xd5\xd4@\x00\xce\tA3\xd5$A\x00|:A\x99\xb9HA3]RA\xcc\xdaXAf\xa6]Af\x02bAf^fA3\x8dkA\x99\xedpAf\x1cvA\xcc\xc2zA\x99?~A\x00=\x80A3;\x81A3q\x82A\xcd\\\x83Af)\x84A\x00\xf6\x84A3\x97\x85A\x99>\x86A\xcd\xfe\x86Af\xcb\x87A\xcd\xb0\x88A\xcd\xa8\x89A\x99\xb9\x8aA\x00\xdd\x8bAf\x00\x8dA\xcd#\x8eA3G\x8fA\x99j\x90A3\x94\x91Af\xd0\x92A\xcd\x12\x94A3t\x95A3\x07\x97A\x00\x94\x98A3\x0e\x9aA\x00\x9b\x9bA\x00M\x9dA\xcc\xf8\x9eAfA\xa0A\x00\xa9\xa1A\x99\x8c\xa3A\xccD\xa5A\x99\xd1\xa6A\x00\xf5\xa7A3\xf3\xa8A\xcc;\xaaAfe\xabA\x99\x82\xacA\x00\x87\xadA\xccx\xaeA\x99j\xafA3V\xb0A\x99Z\xb1A3e\xb2A\xcco\xb3A\x00\x8d\xb4Af\xcf\xb5A3=\xb7A\x00\x8c\xb8A\xcc\xda\xb9A\x99)\xbbAfY\xbcA\x99W\xbdA\x99O\xbeA3;\xbfA\xccd\xc0A\x00\xc0\xc1A\x99F\xc3A\x99\xf8\xc4Af\xa4\xc6A3\x12\xc8Af\x10\xc9A\xcc\xb7\xc9A\x99-\xcaA\xccR\xcaA\xccR\xcaAf\'\xcaA\x00\x1b\xcaA3@\xcaA\xcc\x90\xcaA38\xcbA\x99\x1d\xccAf\x0f\xcdA3 \xceA\x99b\xcfA\xcc\xdc\xd0A3|\xd2A\x00f\xd4A35\xd7A\x99\x8e\xd9A\x00\x8b\xdbA\x99O\xddA\xcc\xaa\xdeA\x99\x9c\xdfA\x99\xda\xdfAf\x96\xdfA3\x14\xdfA\xccl\xdeA3\xde\xddAf\xe4\xddA\x99\xc3\xdeA\xcc{\xe0A\x00S\xe2A\x00b\xe4A\xccj\xe6Af/\xe8A\x002\xeaA\x99S\xecA\x99\x05\xeeA\xccA\xefA\xcc\xbd\xefAf\xb1\xefA\x00H\xefAf\xb9\xeeAf\xb9\xeeA\x00g\xefA\x00\x9d\xf0A\xccH\xf2A\xccW\xf4A\x99\xfb\xf6A\xff\xd0\xf9A\x99O\xfcA\xccd\xfeA\xcd\x17\x00B\xcd\xb2\x00B\x00\x16\x01B3Z\x01BM|\x01B\xb3\x04\x02BMt\x02B\xcd\xe0\x02B\x99u\x03B\xb32\x04B3\x1b\x05B3\x13\x06B\x19\xca\x06B3h\x07B\x00\xde\x07Bf(\x08B\xb3o\x08B\xcd\x91\x08B\x99\x8b\x08B\x80\xa7\x08B\x00\xf5\x08B\x00R\tBf\xbb\tBM4\nB\x19\xaa\nB3)\x0bB\x80\xae\x0bB3!\x0cB\x19{\x0cB\x80\xc5\x0cB\x19\xf7\x0cB\x80"\rB\x19T\rBf|\rB\x19\xb1\rB\x19\x0e\x0eB3\x8d\x0eBf\xf0\x0eB\x19D\x0fBM\xa7\x0fBM#\x10B\x00\x96\x10B\x00\x12\x11B\xe6\xa9\x11B\x99;\x12B\x00\xc4\x12BM*\x13B\x00~\x13B\x00\xdb\x13B\x00\x19\x14B\xcdo\x14B\xe6\xee\x14BLw\x15Bf\xf6\x15B\xe6\x81\x16BL\n\x17B\x00^\x17BL\xa5\x17B\xe6\xf5\x17B\xccO\x18B\x19\xb6\x18B\x19\x13\x19BLv\x19B\x99\xdc\x19B\x80U\x1aB\x19\xe4\x1aB\x80l\x1bB\x19\xdc\x1bBfB\x1cB3\xb8\x1cB\x99@\x1dB\xcc\x00\x1eBf\x8f\x1eBL\x08\x1fB\x19\x9d\x1fB\xcc. B\xb3\xa7 Bf\x1a!B\x19n!B\x99\x9c!B\xcc\xe0!BfP"B\x80\xb0"BL\x07#BLd#B\x19\xbb#B\xe6\xf2#B\xe6O$B\x00\xcf$BfW%B\xcc\xdf%B\xccz&B\xccS\'BL<(B3\x12)BL\xee)B\xe6\x9b*Bf\'+B3\xbc+Bf>,B\x00\xae,B\x00\xec,B\xe6\x83-B\xe6=.B\x80\xcc.B\x99K/B\x99\xa8/B\x19\xf6/B\x99C0B3\x940B\xcc\xe40B\xb3>1B\xe6\xa11BL\x0b2B\x99\x902B\x80G3BL94B3\xd14B\xe6C5B\xb3\xd85B\x99\xae6B\x80\xa37BL\x958B3k9B\x991:B\x19\x8f8B\xe6\xcf3B\x19\xb1,B\x99}!B\x00\xad\x11B\xcc\x83\xfeA\x99\x9e\xd7A\x99\x1c\xb1A\x99\xc0\x8dA\x00\xe8[A\xccz&A3\xbf\x00Af\x1e\xcf@\x99\x91\xae@3c\x9b@3\x8b\x96@3\x9f\xa3@\x00$\xc5@\xcdd\x02A\xcc\x94:A\xcdd\x82Af\xd9\xacAfq\xd8A3\xce\xfeA\xe6E\x0eBf\xfd\x18B\x99\xcb\x1fB\x00\xb8#Bf\xf2%B3"\'B\xb3\n(B3\xd4(B\x99{)B3\n*B\xe6]*B\xcc;*BLr)B\xe6m(B\x00y\'B\xccz&B3\x8f%B\xcc\x8a$B\x80,#B\xe6\n!B\xb3y\x1eB\x19\xdc\x1bB\x00m\x19BfK\x17B\x80?\x15B\x80O\x13BM\x97\x11Bf\x83\x10B\x00\x1a\x10B\xe6\x16\x10B\xe6\x92\x10B3\x94\x11Bf\xb1\x12B\x00\x9d\x13B\x99\xce\x13B\xb3t\x13Bf\xd0\x12B\xb3\x1f\x12B\x19S\x11B\x19<\x10B\x80\xb5\x0eB\x19\x16\rBMK\x0bB\xb3)\tB\xe6 \x07Bf~\x05B3\x80\x04B\x00\xdf\x03B\xcd=\x03B\xb3a\x02Bf\x7f\x01B\x99\xac\x00B\xff\x81\xffA\x99f\xfdA\xff\x06\xfbA3\x82\xf8A\xcc\xac\xf5A\x00G\xf3A\x00\xfa\xf0A\xcc\xa6\xeeA3\xc3\xecA\xcc\xe5\xeaA\xcc\x14\xe9A\x00\x88\xe7Aff\xe5Af\x9d\xe2A3\xce\xdfA\xccU\xddA34\xdbA\x99o\xd9A\x00e\xd8A\x00m\xd7A\x99Q\xd5A\x00\x8d\xd3A\xcc\x12\xd2A3\xab\xd0A\xcch\xcfA\xcc\x13\xceA\xcc\xbe\xccA\x99\x06\xcbA3P\xc8AfO\xc5A\x00<\xc2AfA\xbfA3\xcf\xbcA\x99\xcc\xbaA\x99X\xb9Af\xde\xb7Af\xcf\xb5A\x00\xb4\xb3A3\xe9\xb1A\x00o\xb0A\x99j\xafAf\xc9\xaeA\x00\x7f\xaeA\xcc\xfc\xadA\x99\xc0\xacA3@\xabA\x99\xb9\xa9A\x003\xa8A\xcc\xd7\xa6A\x99\xba\xa5A\x99\xc2\xa4Af\xc4\xa3Afo\xa2A\x00T\xa0A3K\x9eA\xcc\x8c\x9cA\x00\xe1\x9aAf\x98\x99A\x00\xd2\x98A\xcc\x11\x98A\xcc\xdb\x96Af\xc0\x94A\xcd\xdc\x92A\xcdI\x91A3\xc3\x8fA\x00h\x8eA3\x19\x8dAf/\x8bA\x99l\x88A\x99\xc2\x85A\x00D\x83A\xcd.\x81Af\x96~A\xff\xa9zA3\xb1vA\x00nrA\x00\xd4mA3_iA\x00\x1ceA\x99\xf1`A\xcc\xf8\\A3\xe7XA\xcc\xbcTA\x00\x86PA3\x11LA\xcc\xe6GAf\xfaCA\xccd@A\x00&=A\x00>:A\xccn7A\xcc\x025A\xcc\x1a2A3\x85.A3!+A\x99\xc9\'A\xccD%A\x00z#A\xcc\xe0!A\xcc\xaa A\xcc\xf0\x1fAf\xae\x1eA\x00t\x1cA\x00\xca\x19A\x00\xa4\x16A\x00\xbc\x13A\xcdh\x11A\xcd\xfc\x0eA\xcd\xd6\x0bA\x00\x98\x08A\x99\xa3\x05A\x99\xf9\x02A\xcd\xb2\x00A3\xb7\xfd@\xcc\x00\xfb@\xff\xf7\xf8@\x99\xbd\xf6@\x00(\xf3@\x00P\xee@\x00x\xe9@\xcc4\xe5@\xccT\xe1@f\xa6\xdd@3k\xd8@f>\xcb@\x99\xdd\xb4@\xcc\xb4\x97@3\x8bs@3\x836@\x00 \xf4?ff\x88?\x99\xd9\xe3>\x00\x00\xd9=\xcc\xcc\x14<\x00\x00\x00\x00\xcc\xcc\x14\xbc\x00\x00x\xbc\x00\x00x\xbcff\xc6\xbb\xcc\xcc\x14\xbc\xcc\xcc\x14\xbcff\xc6\xbb\x00\x00\x00\x00ff\xc6;ff\xc6<\x00\x00\xf8<\xcc\xcc\x14=\x99\x99-='
'''


def _byte_to_array(data: bytearray, factor=None):
    """
    Converts bytearray containing trace values to numpy array(s). PF6 will return two arrays, min and max array
    Args:
        data: Bytes
        factor: Specific quantity factor to convert integers to floats. Only used for PF6 traces.
    Returns:
        Numpy array(s) with the extracted data
    """
    if factor is not None:
        # pf 6
        data_chunks = frombuffer(data, dtype=int32)
        data_array = data_chunks * factor
        data_array_min = data_array[:len(data_array) // 2]
        data_array_max = data_array[len(data_array) // 2:]
        return float32(data_array_max), float32(data_array_min)
    else:
        # pf 4
        return frombuffer(data, dtype=float32)

def _hex_str_to_array(data: str, num_bytes=4) -> ndarray:
    """
    Converts a hex string containing floating point numbers with size \'num_bytes\' to a numpy array.
    Args:
        data: Hex string
        num_bytes: Number of bytes per number
    Returns:
        A numpy array with the extracted data as floating points
    """
    data_nox = data[2:]
    data_chunks = [struct.unpack('<f', bytearray.fromhex(data_nox[i:i + 2 * num_bytes]))[0] for i in
                   range(0, len(data_nox), 2 * num_bytes)]
    data_array = asarray(data_chunks, dtype=float32)
    return data_array

val = _hex_str_to_array(hex)
val2= _hex_str_to_array(hex2)
print(val2)
plt.plot( val2, val)
plt.show()