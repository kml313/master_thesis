
from datetime import datetime
import pyodbc
from dateutil.parser import parse
import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot
import struct


# abnormal
hex2 = '0x000000005F69954098E3E940D7952941587454418BB48441C0AA99411F25B441AD43C9415A6DE34141B4F841509709426B1214429A4F2142E1DE2B42B7F338422B9743425AD45042754F5B4278786842BF0773424B0E8042EE55854286F48B421332914295C69742380E9D42BAA2A34247E0A842F588AF4282C6B442045BBB42A7A2C0422937C742B674CC424E13D342DB50D84246DBDE42EA22E4426BB7EA42F9F4EF429093F64234DBFB42D0320143ADDB0343ED250743B4C4094300140D43D2B70F4307FD1243D9A0154325F01843E0891B432CD91E43FE7C214334C22443106B27435CBA2A4318542D43599E304335473343608736432726394372753C432E0F3F43645442432AF34443553348431BD24A435C1C4E430DB1504337F15343FE8F564312C65943CE5F5C43F89F5F43922A6243B265654362FA67436B2B6B4327C56D433BFB7043D585734300C676439A507943A3817C4353167F432921814376668243FAFE8343C2418543C1D78643881A88437CAB894344EE8A43B27C8C43F4BC8D43624B8F438E819043770D9243A3439343E5839443D92C954348999543CE9B954375739543FB3B9543ADDE9443FB8D94438C219443F0DA93436099934312769343555A9343334B9343983E934307379343762F9343F12C9343E6279343DA229343CF1D93434A1B93433F16934334119343A30993439804934307FD9243FCF792436BF0924360EB9243D0E392433FDC924334D7924329D292431DCD924312C8924307C3924382C0924377BB92436BB69243E6B39243F1B89243AED492436BF092433F16934381349343CF5793438C739343CF91934307AB9343C4C69343F0DA934328F4934354089443061F9443AD309443DA449443FB5394432868944349779443758B944311989443B8A9944354B69443F0C2944306CD9443A2D99443B8E3944354F09443E4F7944375FF9443060795431C119543AD1895433D209543CE279543E4319543753995430641954396489543AD529543C35C954353649543E46B9543FA759543067B95431C859543AC8C9543C3969543539E95436AA8954380B2954310BA9543A1C19543B7CB954348D39543D9DA9543E4DF954375E7954305EF954396F6954327FE9543B7059643C20A9643D9149643691C964380269643102E9643263896433D429643534C9643E45396437F609643966A9643B7799643CD839643EF929643059D964326AC96433CB69643D8C29643EECC96438AD99643A0E396433CF0964352FA9643740997438A13974326209743C22C9743E33B97437F489743A0579743C2669743687897438A879743B69B97435DAD97438AC19743ABD097435DE797437EF697432508984347179843732B9843943A9843C14E98436860984394749843B6839843689A984394AE9843CCC7984373D99843AAF29843D7069943891D9943B531994367489943945C9943CB759943F88999432FA399435CB7994319D3994346E7994303039A43B5199A4372359A43244C9A43676A9A439E839A435B9F9A4393B89A435BD99A430DF09A43D5109B430D2A9B43D54A9B4392669B43668C9B43A9AA9B43F6CD9B4339EC9B43870F9C43442B9C43924E9C434F6A9C4322909C4365AE9C43BED69C4301F59C43D41A9D4317399D4370619D43B37F9D430BA89D43D4C89D43B2F39D4300179E43593F9E4321609E437A889E4342A99E4321D49E436FF79E434D229F4321489F4385759F43589B9F43BCC89F4390EE9F43F41BA043C741A043B171A0430A9AA043F3C9A043C7EFA043B01FA143FE42A143E872A143419BA143B0CDA14309F6A1437828A2434B4EA243357EA2438EA6A243FDD8A243DC03A343D038A3432961A3431E96A343FCC0A343F1F5A343D020A4434A58A4432983A4431DB8A443FCE2A443F117A543CF42A543497AA543ADA7A54328DFA543060AA6438041A6435F6CA6435FA6A643C3D3A643C30DA743273BA7432775A7438BA2A74310DFA7437F11A843054EA843EE7DA843EEB7A843D8E7A843E326A943CC56A943D795A943C1C5A9434602AA433032AA43B56EAA4325A1AA4330E0AA439F12AB43AA51AB431984AB43A9C5AB439EFAAB432F3CAC432471AC432EB0AC4323E5AC43B426AD43A85BAD43399DAD43A8CFAD433911AE43A843AE433885AE432DBAAE43BEFBAE43B230AF43C974AF43C8AEAF4364F5AF43592AB0436F6EB04364A3B043FFE9B0437A21B1430A63B143849AB1439BDEB1439A18B243BC61B2433699B243D2DFB2434C17B3436D60B343F29CB34314E6B3439922B443BA6BB44340A8B443E6F3B443E62DB5430877B5438DB3B543B901B6433F3EB643E589B643E5C3B6431212B7431D51B743499FB74354DEB743802CB8438B6BB8433DBCB843C3F8B8435E3FB943AC62B9435E79B943D876B943AC62B9436944B9438014B94310E2B843759BB8435F57B8431CFFB743FBB5B7433E60B7436B3AB743492BB7435F35B743974EB743546AB7431C8BB74354A4B74397C2B74349D9B743FBEFB743A201B8434913B8436A22B8438B31B8431C39B8433243B843C34AB8435452B8435F57B843EF5EB8437561B843FB63B843FB63B843FB63B8438066B8438066B8438066B8438066B8438066B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B843FB63B8437561B8437561B8437561B8437561B8437561B8437561B843EF5EB843EF5EB843EF5EB8436A5CB8436A5CB8436A5CB843E459B8435F57B8435F57B843D954B843D954B843CE4FB843494DB843C34AB8433D48B8433243B843AD40B843A23BB8431C39B8431134B8438B31B843802CB8437527B8436A22B843E41FB843D91AB843CE15B8433E0EB8433209B8431CFFB74311FAB74380F2B743F0EAB743D9E0B74349D9B74333CFB74327CAB74311C0B74381B8B7436AAEB7435FA9B743499FB7433E9AB7432890B7439788B743FB7BB7436A74B743DA6CB7434965B743B85DB7432856B743974EB7430647B743763FB743E537B743CF2DB743C428B7433321B743281CB7439714B743070DB7437605B7436B00B743DAF8B64349F1B643B9E9B643AEE4B6431DDDB64312D8B64381D0B64376CBB64360C1B64355BCB643C4B4B64333ADB6431DA3B6438C9BB6437691B6436B8CB6435582B643C47AB643AE70B6431D69B643075FB6437657B643604DB643CF45B643B93BB643A331B6438D27B643FC1FB643E615B643550EB6433F04B64329FAB54398F2B54382E8B5436BDEB54355D4B543B9C7B54329C0B5438DB3B54377A9B543DB9CB5434A95B543AE88B5431E81B5438274B543F16CB543DB62B5434A5BB5433451B5431E47B5438D3FB543FD37B5436C30B543DB28B5434B21B5433F1CB5433417B5432912B5431E0DB543980AB5431308B5438D05B5430803B5438200B543FDFDB44377FBB443F2F8B443F2F8B4436CF6B443E6F3B44361F1B443DBEEB44356ECB443D0E9B4434BE7B443C5E4B44340E2B443BADFB44334DDB443AFDAB44329D8B443A4D5B44399D0B44313CEB4438ECBB44308C9B443FDC3B443F2BEB4436CBCB44361B7B44356B2B443D0AFB443C5AAB443BAA5B443AFA0B443A49BB4431394B443088FB4437787B4436C82B443DC7AB443D175B443406EB443AF66B4431E5FB4438E57B443FD4FB4436C48B443DC40B443D13BB4434034B443AF2CB4431F25B4431420B443081BB443FD15B4436D0EB4436109B4435604B4434BFFB34340FAB34335F5B343AFF2B343A4EDB34399E8B34314E6B3438EE3B34383DEB343FDDBB34378D9B343F2D6B3436DD4B34362CFB34362CFB343DCCCB34357CAB343D1C7B343D1C7B3434BC5B343C6C2B343C6C2B343C6C2B34340C0B34340C0B34340C0B34340C0B34340C0B34340C0B34340C0B34340C0B343BBBDB343BBBDB343B0B8B3432AB6B3431FB1B34399AEB3438EA9B34383A4B343789FB3436D9AB3436295B343DC92B343'
hex ='0x664B9240666AA0404DF79540F3D0A240331CA240B385A3404D62AF40E6A6A440CDF8AD40E679A7401A9BA8406689AE40F3A3A5408D53B440CD25AB405A32B3400093B7409AAAAF40E63EBB404D62AF4040B1B9408029B640F3C2B340000CC040662FB4401A06C240C01ABB409AC9BD40A66CC440F395B640CD90C440A620B9401A60BC40E6E4C040A6C6BE404D73CE4000DFC2404DA0CB404DCDC8400085C8408094CF40DAE7BF401A7FCA4000DFC2406621C5401A7FCA40402AC2405A43D2409A42C640334CCF40CDAFD2403379CC40A604DB408094CF403398DA40C058D7409A61D44000F0E140E622DD40B3F3EC40B37AE4406605E740335DEE409A26E840DA36FB40C03CF94000BD044173290541B374044100360D4140AE094133FD1341E0CC11410DAC1641FA121F410D251F41B3FE2B4153FE25412D802B41CD522841A62E2841A6D42D41CD252B4180293641B3773441C6D53F4126304041E6B74341203A4F41F32D4D4160A45C41B3D45E415A816E4100B57541A6BB7F4153ED86413A998A41B366954136C49741B679A44153FEA541ADBCAF418611B841C6B6B14150FCB541B3FEAB41B0EB9C41E0538941A6AA6041062F2E41B3A10141C0EDBD401AD6944066247540B35B5640B33C4840801B4740669A4D4033C5574033E465401A097240333880404DD88740DA118D4080729140C09093408045944073489340DAB792408D9C8F40CD148C4033B1884033DE8540330B83401A118140CDDF7F400D1480405A5C80404D328240DA98844080268640C0718540B347874040739D409A23B8409A07DA40DA36FB4033840B410DAC16417A7C2041A62E2841A6D42D41C0A13241E6983541E66B384113783A412D723C4100393D41D3FF3D41263040419A424641334C4F4160D159415A35634100696A416D936E415A5471415A27744186067941FA187F414D328241AA6B84413DE7854166FF864130BD87412AA88841BD238A412A7B8B41BDF68C41E60E8E41CDE78E41BDC98F4186879041463C9141B3939241E6B4934106C4944160099641BA4E974160DC9841E0459A4160AF9B4153859D4120379F41D0CDA041AD9DA141A688A241532BA3417D43A4418A40A5418D34A64126C5A64116A7A741FD7FA841E358A9410D71AA411A6EAB4173B3AC4176A7AD418DADAE41C0CEAF411A14B141866BB2413602B441AD62B541A038B741F368B9417DDBBA41367BBC416093BD41335ABE411A33BF41C6D5BF416066C0410D09C141BAABC1417057C24113F1C2416039C34120EEC341A66CC4415D18C5414DFAC5416D09C741B33CC8410D82C941FD63CA41267CCB413D82CC412D64CD414D73CE413D55CF412D37D041B3B5D04100FED041CABBD141634CD24106E6D241966DD341EDBED341C085D4419D55D541C66DD641CA61D741800DD8411095D8411389D941A310DA4146AADA418AE9DA41237ADB41BD0ADC41ADECDC416398DD41A0C2DE41869BDF419398E041534DE1414D38E241E6C8E241C398E341E3A7E44120D2E541EA8FE6410096E741E66EE84133B7E841A623E941067EE941D03BEA41CA26EB4193E4EB415DA2EC41A0E1EC41437BED41CAF9ED41638AEE412D48EF41DAEAEF410303F1419D93F141967EF24103D6F3414000F541BD75F5417D2AF641D37BF64116BBF641069DF7411DA3F8418606F941C03CF941168EF941F35DFA41ED48FB415097FC41D000FE4190B5FE41C3D6FF4165DF00424BB80142866802426BC70242E6C2024288E202428053034295DF034236FF0342703504422E70044260170542630B064210AE06427A1107426AF30742C2BE0842D64A09422D9C094283ED0942B31A0A42B5940A426AC60A4273CF0A42DD320B4218E30B42F5B20C42265A0D42A3CF0D42FB9A0E42FE8E0F42981F1042DB5E1042BE431042CAC61042504511423D3311428A7B1142F3DE1142DAB71242DDAB1342763C1442AA5D15424568164280181742EA7B1742F384174270FA17421B23184270FA17423AB818429B8C1942F5D11A42683E1B4250911C42AAD61D424D701E420BAB1E42ADCA1E4263761F4293A31F42B0BE1F42C54A204228992142C3A32242A0732342C2FC24421AC82542123926428D342642A2C0264298B7264230CE2642E6792742E0642842F66A2942DD432A42AAF52B4232EE2C42033B2D42D30D2D429DCB2D4280B02D428AB92D42A84E2E42D3E02F423EBE304285F13142FB513342A8F43342662F3442AA6E34420AC9344242853442154C3542BA5F364213A537422AAB3842EECD3A4220753B42035A3B42AE823B426DBD3B426B433B4235013C4290C03D42C3E13E423BBC4042C52E4242BD9F42427A60424226034342AA8D42421DFA42426D364442D0844542761247427DFA4842D34B4942C35A47428BCB4442A8134242B6E43D429A23384278F43042787B2842031C1F42964B1542250D0A4256ACFB41B059E641D3C4D14126A9C841B048C741F64ECB41331FD2419A07DA41F6E6E14140B4E9418078F1417000F841669DFD41362C0142805303429A4D0542B2CD0642A3290842CD410942C62C0A42A3FC0A425AA80B4248100C42F3380C42EA2F0C423E070C428AD50B42B8880B42320A0B420A6C0A427AE40942EA5C094233B10842A329084298A607425DF60642B053064275A3054243FC04421255044265B203429BF402427D5F0242D0BC0142FDF5004220260042B6D9FE41EA27FD4160B5FB419D0CFA41C651F8410DB2F6415312F5410DDFF3413624F241D3D5F041405AEF41C0F0ED412D75EC412A81EB4166D8E941AD38E8417A17E7416D1AE641BD83E441936BE3413A26E241EAE9E0410311E041D0EFDE41BAE9DD41266EDC41E03ADB41AD19DA41AA25D941CD55D8414DECD641CD82D5413A07D4412301D341C0B2D1413640D0416085CE41EA24CD41ADFACB412388CA41AD27C94136C7C7419A42C6411AD9C441431EC3419D90C1417084BF414378BD41F047BB4146C6B84150FCB5414620B3416D71B041FD25AE4153A4AB41C63DA9413AD7A6416A31A441B6A6A141D3EE9E4163A39C4180EB9941BA4E9741C3849441BAA89141E0F98E4166A58C4100638A4113A2874190448541CAA7824163658041ADFD7B410DB27641402D724173A86D4166D86941D3896541CDCE604166B95B4173105741401C5341133D4E41F35A4A4100B24541A69941413A6F3D41BA323941FAAA35411A14314160A12C41CD5228417A4F234133491F41CD331A41661E1541C0BD104166A50C41AD3208411AE403410D14004166F7F7403330F140B320EA400DEDE240004ADC40A65ED5409ABBCE4026A9C8404D27C340DA14BD401ABAB640B3A4B14066B6AB408D34A6400D259F40E65A99408D6F924080CC8B4066FF86400D148040CD9374409A266840668C5E4000D1534080EE4940000C404066D53940330E33401A6E2B40B3582640CDAC224066971D40B30C1B4033A31940CD60174000AF15404DCA184066C41A4066971D40663D2340B32B2940B3FE2B40B3D12E4000C03440CD713640B34A37409A233840660237401AE7334000ED314066892E4066E3284033EF244000FB20404D9D1B409A3F1640CDBA1140CDE70E40CD41094033DE054000EA0140CD39FA3F000FF03FCD47E93F3311E33F66B9DB3F00FED03FCD36CA3F9A15C93FCD90C43F0066BA3F00C0B43FCDF8AD3F001AAF3F9A31A73F33499F3F33D0963F00DC923FCDBA913F66D2893F662C843F00EA813F33A9793F9A45763F9A45763F3303743F9AF96A3F9A53653F00F0613F00F0613F00F0613FCD74663F33B7683FCD74663F003C6D3F9AF96A3F667E6F3F9A45763FCD1A6C3FCD1A6C3F9AF96A3F0096673F9AF96A3F33B7683F33B7683FCDCE603F3311633F00F0613FCD74663F00F0613F6632643F0096673FCD74663F003C6D3F33B7683F335D6E3F33B7683F667E6F3F335D6E3F667E6F3F9A9F703F9AF96A3FCDC0713F00E2723F334F7F3FCD0C7D3F0090873F9AF38A3F00638A3F66D2893F66D2893F330B833F0088783F00F0613FCD90443F333B303F00DC123F00F0E13E333BB03E9AF96A3E66B62B3E9AF9EA3D'

'''
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
        data_chunks = np.frombuffer(data, dtype=np.int32)
        data_array = data_chunks * factor
        data_array_min = data_array[:len(data_array) // 2]
        data_array_max = data_array[len(data_array) // 2:]
        return np.float32(data_array_max), np.float32(data_array_min)
    else:
        # pf 4
        return np.frombuffer(data, dtype=np.float32)

def _hex_str_to_array(data: str, num_bytes=4) -> np.ndarray:
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
    data_array = np.asarray(data_chunks, dtype=np.float32)
    return data_array

val = _hex_str_to_array(hex)
val2= _hex_str_to_array(hex2)
print(val2)
plt.plot( val2, val)
plt.show()