
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

'''
abnormal
hex2 = '0x000000004ACD7C400213BE40AD6EFD40D33C1E4102133E417F985D4154467D41E9658E41A7289E4192FFAD41F799BD41E270CD41A033DD4132E2EC41F1A4FC41C129064237150E4269E21542B2B91D42129B25425B722D42A3493542EC203D421FEE44427ECF4C42B19C5442E4695C422D41644249046C42A8E57342C5A87B42E6B081427F9785420D79894290558D42133291428B0995421AEB984292C29C42FE94A042766CA442CD34A8425011AC429CD4AF42E797B3424965B742A02DBB42E0EBBE422CAFC2426168C642A126CA42CCDACD42EB89D1420A39D54229E8D842328DDC423A32E04243D7E3424177E7423F17EB421BA8EE420E43F242EAD3F542A555F942294AFC42E46FFE42AF0100430864004356870043E68E00433F7D00436C570043F21F0043A1ADFF422702FF428C47FE42FB91FD42A3F5FC428272FC4277F9FB42A399FB42F248FB425602FB42D1C5FA42568EFA42F260FA429938FA424010FA42FEF1F942BBD3F94283BAF9424CA1F9421F8DF942E873F942BB5FF9429A50F9426D3CF9424C2DF9422B1EF9421414F942F304F942DDFAF842BBEBF842B0E6F8429ADCF84284D2F8426EC8F84257BEF8424CB9F84236AFF84220A5F8420A9BF842F390F842DD86F842C77CF842B172F8428F63F8427959F842584AF842363BF842152CF842FF21F842DD12F842C708F842B1FEF74290EFF74279E5F74263DBF7424DD1F7424DD1F742A6F9F7424D45F842FE95F842C7F0F842A555F94283BAF9424B15FA421F75FA42DCCAFA428E1BFB423467FB42C5A8FB4255EAFB42E62BFC426B68FC42E69FFC4255D2FC42C404FD423337FD429764FD420697FD4276C9FD42DAF6FD42331FFE428042FE42CE65FE421C89FE425FA7FE42A1C5FE42E4E3FE421CFDFE425316FF428B2FFF42B743FF42E457FF420567FF42327BFF42538AFF42809EFF42ACB2FF42CDC1FF4205DBFF4231EFFF42AF010043C50B0043DB150043F21F00438227004399310043343E00434B48004361520043FD5E0043986B004334780043D0840043E68E0043FC98004313A3004329AD00433FB7004355C100436CCB004382D5004398DF004334EC00434AF6004360000143770A01438D140143A31E0143342601434A300143E63C0143FC4601438D4E0143A3580143B9620143CF6C0143E6760143FC8001438D880143A3920143349A01434AA4014360AE014376B8014312C5014328CF0143C4DB014360E80143FCF4014397010243330E0243CF1A02436B27024307340243A24002433E4D0243605C0243FB68024397750243B884024354910243F09D02438CAA024328B70243C3C302435FD00243FBDC024397E9024333F6024354050343F01103438C1E0343272B03433E350343D9410343754E0343115B0343326A034354790343758803439697034332A40343D9B5034380C7034327D9034353ED0343800104432713044353270443803B0443AC4F04435E660443107D0443489604437FAF044331C604435EDA044310F10443C2070543741E054326350543D84B05438A620543C27B054374920543ABAB0543E3C405431BDE0543D8F905430F13064352310643954F0643526B0643958906435DAA064325CB0643EDEB0643300A0743F82A0743C14B0743896C0743CC8A07430EA9074351C7074319E80743E2080843AA290843F84C0843C06D08430E910843D6B1084324D50843F7FA0843CB20094393410943666709433A8D09430DB30943E0D809432EFC094302220A43D5470A43A86D0A437C930A434FB90A4323DF0A4370020B4344280B439D500B4370760B43BE990B4317C20B4370EA0B43C9120C43A73D0C430B6B0C43EA950C434EC30C43B2F00C43901B0D43F4480D43D3730D43B19E0D4390C90D43F4F60D43DE260E4342540E43B1860E4320B90E438FEB0E43791B0F43624B0F434C7B0F4336AB0F4314D60F4378031043DC3010434C631043B090104399C0104308F31043F222114361551143D087114340BA1143AFEC11431E1F124313541243828612436BB61243DBE81243CF1D13433F501343AE821343A2B7134312EA1343061F1443765114436A8614435FBB144354F01443C3221543B8571543AC8C15431CBF154396F61543963016431068164395A4164395DE16439518174395521743958C17438AC117438AFB174389351843896F18430FAC18431AEB18439F27194324641943AAA01943AADA1943AA141A432F511A433A901A433ACA1A4345091B4350481B435B871B43E0C31B4366001C43713F1C437C7E1C4386BD1C4391FC1C43223E1D432D7D1D43BEBE1D434E001E43DF411E43F5851E430BCA1E439B0B1F432C4D1F43C7931F4358D51F436E1920430A602043A5A6204341ED204357312143F377214309BC214399FD2143354422434B8822436CD122438E1A2343296123433FA5234355E923436C2D244307742443A3BA24433E012543604A254381932543A2DC2543C32526436A71264311BD2643AD032743D9512743FA9A274326E92743CD34284374802843A0CE2843C1172943686329430FAF294330F829435D462A430E972A433BE52A4367332B4393812B433ACD2B435B162C4388642C432EB02C4366032D4392512D43BF9F2D4370F02D439D3E2E434F8F2E437BDD2E4338332F43EA832F43A7D92F43592A3043907D30434DD33043FF233143BC793143F4CC3143362532436E7832432BCE324362213343147233434BC5334383183443C570344308C934434B213543087735434ACF354307253643CF7F364397DA3643E53737433395374380F23743494D384311A838435300394326603943EFBA39433C183A438A753A43D8D23A431A2B3B43ED8A3B43B6E53B4303433C43D6A23C4324003D43EC5A3D433AB83D4302133E435B753E43B4D73E43923C3F43659C3F43BEFE3F430C5C404365BE4043432341439C854143F5E74143C84742439BA74243F4094343D26E4343A5CE434384334443DD954443BBFA44438E5A45436DBF45434B244643AF8B464398F54643FC5C474360C447433E294843AE95484397FF484380694943F0D54943533D4A433DA74A43AC134B431B804B438AEC4B4373564C43D7BD4C43C1274D4330944D4325034E4319724E430EE14E4302504F43F7BE4F4371305043EBA15043651351435A825143D4F35143C962524343D45243C84A534342BC534342305443379F5443B110554336875543B0F855432A6A56432ADE5643A44F5743AFC85743343F5843BAB55843C42E59433FA059433E145A43498D5A43CE035B4349755B4348E95B43485D5C4353D65C435E4F5D43E3C55D4373415E43F9B75E4303315F4394AC5F439F256043A99E60433A1A6143459361435B116243718F6243860D63430C8463439CFF6343B27D644342F96443D3746543DEED6543F46B664315EF6643B06F67434CF06743E770684383F16843A4746943C5F7694360786A43FCF86A4312776B4333FA6B43547D6C4375006D438B7E6D4327FF6D43C27F6E4369056F4304866F43A00670433B8770435C0A71430390714324137243BF937243E1167343029A7343231D7443BE9D74435A1E75437BA175431622764337A576434D2377434D977743BC0378430A617843C7B67843D2F57843C62A79438F4B7943365D7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F7943BB5F79439A5079431F197943DDC078436E5478436EE07743F46E77436FF876437A8976430B1D7643A7B575434E53754301F67443BE9D74430C4D7443E0FE7343BFB573439E6C73430D2B73430DF172430DB772439E847243B45472435027724372FC71439FD6714351B371437D8D71439F6271433B357143CC027143D7CD7043'
hex ='0x00A9764000BC874080B09E4000AD6E4080D1A54000978840806E90400085AC400008824000CCB04080429F40809193408033BD4080268E4080CAB340005FAF4000DA94400053C74000B19D4000CBB2400031C24080D69B4080BDCD400084AE4000EDB740804DD2408018AA400092DB40002EC84080C2C340801AEF400009C94000CEF5400017F6408020E34080DD0D4180CDF640004E0841C034164180BD04418041214180201A4180FC1841805C3441C0E5214100813441C0FA4041C04B31414085504180BE4B41C0CF4D4180656B4180485C4100F472418051814100F3744120E88A4180DE8B410072894180699A41E0AB954180209A416034A941E0CAA041C02AAA410038B441E0C3AE41C04DAD414026B341809EC2410032C0416007BA41C093B341C02AAA4140309F41E0639341C0F385410063704180705541807C3D4180F42841008F184140CD094180F0F9408044E4400029D24080C3C1408037B5408018AA40801DA040008F98400000924080958B40002B85408051814000157A4000F66E4000D5674000B55E4000255A40002756400097514000E24D4000074D4000764A40002D4A4000514B4000BD4E4000DD5740006C5E400020644000D46940003E714000A878408076804080508340002B8540004F8640804E874000E18640802A864080BC864080E1854000748540002C83408076804000827B4000A8784000F3744000886F4000F66E40005E7A408004894080FA9C4000CBB24000E3CB4000D6E54000A5FE4080950B41006C154180B01E41C075264180A92C41404B324100A4374140FD3B41C01F404100794441C088494140F44D4100DF534100255A4100D86141C02F694180F56F418017754100F17841C0937C4120128041809A814120358341E0D884410098864180728841C0A8894140A88A4180958B4120558C4180028D41601D8E4100948E4180018F4120788F41C0EE8F410093904100259141C0A49141601B924160AD924180489341E0F5934180B59441603E954100B59541A0749641006B9741A07398412073994120979A4160849B4180689C4160839D4140E79E41E0A69F41409DA04140C1A1412093A241A092A341A06DA441A048A541A023A64180F5A64160C7A7414099A84100F4A941E00EAB4120B3AB4140E0AC4160C4AD4140DFAE4120B1AF414095B041A042B141600BB24120D4B24120AFB341E077B441201CB5418012B64180EDB641C023B841E050B941C06BBA4180C6BB4180EABC414045BE4100A0BF4140D6C04180C3C141408CC241A082C3414042C44140D4C441802FC54140AFC541E025C641C065C64140D3C641E000C74180E5C6410053C74140F7C741A05BC84100C0C841E048C941E091C9418008CA41C063CA416091CA4100BFCA412011CB41A07ECB41A0C7CB4100E3CB416047CC4100BECC41A034CD41A07DCD418006CE41207DCE4180E1CE412058CF41A00ED04180E0D041E044D14140A9D141E0B1D241004DD3418003D44120C3D441004CD541C0CBD541E0D4D5414039D641608BD641C0EFD641C038D741806FD741C0CAD7414038D841804AD841804AD84180DCD841200AD94180B7D9418049DA41A02DDB41C07FDB4140EDDB416088DC41C0ECDC412051DD4160F5DD410023DE41202CDE41C010DE4120E3DD4160F5DD4160F5DD41C010DE41E062DE41C034DF41C00FE04120BDE041A02AE141C0C5E1416085E24100FCE241C07BE341A0BBE3410020E441C09FE441C031E541C0C3E541E05EE64180D5E64180B0E741C09DE8410042E941E038E941400BE941801DE9418066E94140E6E94160EFE94160EFE941A001EA41C09CEA414053EB4120DCEB41402EEC41A000EC41001CEC41A049EC4180D2EC41E036ED41C008EE4140BFEE414051EF41C007F041A022F14100D0F1414074F241E033F34160EAF34140BCF441208EF5412069F64100F2F64160C4F641E09FF641C0DFF641C028F74120D6F74180F1F7412068F84100A8F84100F1F84140BAF8414028F8418083F84100A8F841208CF94100A7FA416054FB41201DFC4120AFFC414093FD41A0F7FD41E052FE410080FF41704D004220ED0042907A014260DA0142C0F5014220C8014280E3014250FA0142303A0242B05E0242009A0242C0D00242909E0242A07E0242B05E0242B0A70242B0390342E0B40342D01D044210790442D0F80442002B0542C0AA054240610642F0000742A0570742608E0742207C0742F0DB07429009084200050842E0FB074230370842200E084270B70742608E0742D0D20742F06D0842301209428096094220C40942601F0A42A07A0A42B0350B42A0300C4220E70C4290740D4200700D42D03D0D42F0460D42B07D0D42E0AF0D4270B40D42C05D0D42F0FD0C42D0F40C42D03D0D4270FD0D4290980E42C0130F4230A10F42D0171042009310425060114210E01142A02D12424012124290041242503B1242C03612420049124240121242A09B1142B07B114250F2114210BB1242407F134280DA1342E03E1442701E154200FE15421002174250A61742B078174200221742D038174290261742603D1742A04F174260AB164250391642B0541642202B174280D81742806A1842C00E194230E5194230C01A42C09F1B42703F1C42801F1C4270F61B42A0281C42F01A1C42E0F11B42C0561B42E0CD1A4220E01A4200B21B42204D1C4210B61C4210481D42A0271E42F0861F42106B2042908F2042F0182042E0382042C0782042C0782042E0EF1F4260F01E42409E1E4260391F42A06F204230062142C09C2142D057224200652342605B2442703B2442B0BB234250A02342C052234240E52242302A224260132242209322424077234220B72342D0562442D07A2542F0F0264250E72742D0C227425055274200632742206C274260EC2642A0DA2542304D2542A023264260EC2642108C274230272842B0262942404F2A4280AA2A42F05C2A4290F829428086294200192942A06B284220FE2742B0DD2842F0CA2942404F2A4230012B4200852C42F0C82D42D0BF2D42C0042D42208E2C4210652C42C0972B42F05C2A42E07C2A4230932B42F05B2C4200CE2C42B0B62D4230482F42A0D52F42B06C2F4270112F42A0B12E42E0E82D42901B2D4260322D4230242E42707F2E4200162F42C0B93042106232421062324210D03142C09431428039314280832F42609F2E42B0B52F42A0B03042A08B3142A0AF324240933442001335424093344210183442B0B333421019324210F53042B0B4314240DD324270A13342E0093542500437426076374220D23642C0B63642E0BF36428037354200EF334220D3344230203642D0DF3642F0553842105E3A4250703A4220F53942F0C2394240DA3842D0283742D0DF3642A06338427055394210A73A4250013D42B0F73D42B0653D4290133D42E0BC3C42C0FD3A4200EC394250023B42A0183C42D0253D42905B3F4290114142809F4042301B404220A93F4240453E42807C3D4290803E42B0643F422084404260954242C0D44342C0F94242506C4242E0DE414240FB3F4290A43F42A03A41421011424240D54242B0184542809C4642A013464290A14542F0E1444290104342105A424280794342702B444250B444427006454220144542A05D444230F54242F02C4142A03B3F42807C3D42E0E13B42D0943A42705539420036384240243742E02D3642103C354250733442201D334230DA2F42E0572B4200D22442409F1C42309F1342F06C0A42E0B501424099F141A006E041C0CECF4180E8C0414070B14180F7A24160AB964100BA8B4100768141C050704180FE5D4100BE4C41406A3D4100F12F41C09B2341400F184140A70C4140F5024100F4F2408046E04080BECB4000EFB34000D79A40'

'''
hex2 = b'\x00\x00\x00\x00F\x93\xdd@\xce\x82%A)\x07\\A(?\x89A\xe3\xe2\xa3Ajy\xbeA\x1c\x99\xd8A\x00\xa5\xf2A\x07\x1d\x06B\xc0\xd3\x12B\xdcb\x1fB\xc2\xc3+B\xf3\x068B\xa8\'CBS\xfbJB\xdeuRB\xab\xa6VBo\x8aWB$\xa8WB/\xd8VBV\xcbTBANRBC\xa6OBt\xc9LB\xf2\xefIB\x8c-GB\xae\xdeDBt\x1aCB\xda\xbfABa\xbe@Bn\x0f@B\xe6\x9b?B.]?B\xc6B?B,<?B\x13F?B\xe1Y?BIt?B\xff\x91?B\x01\xb3?B\x04\xd4?B\x06\xf5?BU\x19@B\x0b7@B\xc0T@Bur@B\xde\x8c@BF\xa7@B\xae\xc1@B\x17\xdc@B2\xf3@BM\nAB\x9d.ABkBAB9VAB\xd6}AB\x07jAB\xa4\x91AB\xbf\xa8AB\x8d\xbcAB\xa9\xd3AB\x11\xeeAB\xdf\x01BB`\x12BB\xe2"BBc3BB\x97@BB\xcbMBB\xffZBB3hBB\x1brBBO\x7fBB\x83\x8cBB\xd2\xb0BB\xdeUCB\x89CDB\x05fEB\xb4\x95FB\xcc\xdfGB\x97&IB+?JB\xf2CKB\x81\x1aLB\x8d\xbfLB\xe4FMBl\xbaMB\xa5\tNB]HNB\xe0yNB0\x9eNB\xe5\xbbNB\x9b\xd9NB\x19\xc9NB\xb6\xf0NB\xb6\xf0NB\x03\xf4NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NBP\xf7NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\x03\xf4NB\xb6\xf0NBi\xedNB\x1c\xeaNB\xcf\xe6NB\x19\xc9NB\xb3\xcfNBM\xd6NB5\xe0NB\x82\xe3NB\x19\xc9NB2\xbfNBK\xb5NB\x17\xa8NB0\x9eNB\xaf\x8dNB-}NB_iNB\xdeXNB]HNB\x8f4NB\xc0 NB\xf2\x0cNBq\xfcMBV\xe5MB\x87\xd1MB\x06\xc1MB\xeb\xa9MB\x1d\x96MB\xb4{MB\xff]MB\x97CMB\xe1%MB\xdf\x04MBw\xeaLB\xc1\xccLBY\xb2LB\xf1\x97LB\x88}LB\xd3_LB\x84;LB4\x17LBK\xecKB\x14\xbeKB\x90\x8cKBsTKBU\x1cKB\xea\xe0JB\x1a\xacJBIwJB+?JB[\nJB\x8a\xd5IB\xba\xa0IB\x83rIBLDIB\xc9\x12IB\x92\xe4HB\x0f\xb3HB\xd8\x84HB\x08PHB\xb6\nHB\xfe\xcbGB\x12\x80GB&4GBS\xdeFB\xcd\x8bFBG9FB\xc1\xe6EB\xd5\x9aEB\x83UEB\x98\tEBF\xc4DB\xf4~DB\xa29DB\xea\xfaCB\xe5\xb8CB\xe1vCB\x8f1CBV\xe2BB6\x89BB\x160BB\\\xd0AB<wAB\x1c\x1eAB\x14\xbb@B(o@B\xef\x1f@B\x04\xd4?B\xb2\x8e?B\xadL?B[\x07?B"\xb8>B\xe9h>B\xc9\x0f>B\x90\xc0=B\xbdj=B\xea\x14=B\x17\xbf<B\xf7e<B$\x10<BQ\xba;B1a;B\x92\x18;B@\xd3:B\xee\x8d:B\xeaK:B2\r:B\xe0\xc79B\xf4{9B\x0809B\x1c\xe48B~\x9b8B\xdfR8B\xda\x108B"\xd27B\xb7\x967BM[7B|&7B\xab\xf16B(\xc06B\xa4\x8e6B\xbbc6B726B\xb4\x006B\xe3\xcb5B\x12\x975BBb5B\xbe05B:\xff4B\xb7\xcd4B\xff\x8e4B|]4BE/4B\xc1\xfd3B\xd8\xd23B;\xab3B\xec\x863B\x9db3BM>3B\xb1\x163B\x14\xef2B\xc5\xca2B\xc2\xa92BZ\x8f2B?x2B#a2B\x08J2B\xed22B\x85\x182B\xcf\xfa1B\x1a\xdd1Be\xbf1B\x15\x9b1B\xc6v1BvR1B\xda*1B\xf0\xff0BT\xd80Bj\xad0B\x81\x820B\x97W0Ba)0B\xdd\xf7/B\x0c\xc3/B<\x8e/B\xb8\\/B\xcf1/B\xe5\x06/B\xfc\xdb.B_\xb4.Bv\x89.B\x8c^.B\xa33.B\xb9\x08.B\x83\xda-B1\x95-B\xadc-B\xf5$-B>\xe6,B9\xa4,B4b,B\xc9&,B\x12\xe8+B\xa7\xac+B\x89t+B\x06C+B\xcf\x14+B3\xed*BI\xc2*B`\x97*B)i*B@>*B\xf0\x19*B\xee\xf8)B\x85\xde)Bj\xc7)B\x9c\xb3)B\x81\x9c)Be\x85)B\xb0g)B\xaeF)B\xab%)B\\\x01)Br\xd6(B\x89\xab(B\x9f\x80(BiR(B\x7f\'(B\x96\xfc\'B\xac\xd1\'B]\xad\'B&\x7f\'B=T\'B\xed/\'B\x9e\x0b\'B\x83\xf4&B\x99\xc9&B\x8d$&B/:%BK\xfd#B/q"B\x0b\x82 B\x97n\x1eBS&\x1cBA\xca\x19B\xb0~\x17Bl6\x15B_\x1c\x13Bm\x19\x11BKK\x0fB\x16\xc9\rB2\x8c\x0cBl\x87\x0bB,\xd5\nB\x8d\x8c\nBy\xd8\nB\xb9\x8a\x0bB\x95d\x0cB%;\rB~\xe3\rB\xb9S\x0eB\tx\x0eBSZ\x0eB\xb4\x11\x0eB\xad\xae\rB\x8b4\rB\x00\xa0\x0cB\\\x15\x0cB\xd2\x80\x0bB`\xe2\nB\x88J\nB\xe2\x9e\tB\xd6\xf9\x08Bd[\x08B\x0b\xb3\x07Be\x07\x07B\xbf[\x06B\xb3\xb6\x05B\xf4\x14\x05B\x1a\\\x04BX\x99\x03B1\xdd\x02B\xf1*\x02BK\x7f\x01B?\xda\x00B\xcd;\x00B\x1fU\xffAm\x04\xfeA%\xef\xfcAx\xe0\xfbA\xcd\xf2\xfaA\x8b\x1f\xfaA\xe2R\xf9A\xd4\x8c\xf8A)\x9f\xf7A~\xb1\xf6A9\xbd\xf5A\xf5\xc8\xf4A~\xe8\xf3A\x07\x08\xf3A(\r\xf2A\x13\xe4\xf0Ae\xd5\xefA\x1e\xc0\xeeA\xd7\xaa\xedA\x92\xb6\xecA\xe7\xc8\xebAp\xe8\xeaA.\x15\xeaA\xb74\xe9A\x0fh\xe8A\xcc\x94\xe7A$\xc8\xe6A\x15\x02\xe6Am5\xe5A-\x83\xe4A\xea\xaf\xe3A\x0e\xd6\xe2A3\x1d\xe2A%W\xe1A\xe5\xa4\xe0A\xd9\xff\xdfA3T\xdfA\xc1\xb5\xdeA\xb5\x10\xdeA\x0fe\xddA\x9d\xc6\xdcA+(\xdcA\x88\x9d\xdbA~\x19\xdbA\xda\x8e\xdaAk\x11\xdaAa\x8d\xd9A\xf2\x0f\xd9A\x82\x92\xd8A\xdf\x07\xd8A\t\x91\xd7Ah\'\xd7A-\xb7\xd6A\x8bM\xd6A\x1c\xd0\xd5AFY\xd5A=\xd5\xd4A3Q\xd4A\xf8\xe0\xd3AWw\xd3A\x84!\xd3A\xe5\xd8\xd2A\x12\x83\xd2A?-\xd2A:\xeb\xd1A\x9b\xa2\xd1A\xcam\xd1A\xc8L\xd1A,%\xd1A]\x11\xd1A[\xf0\xd0A\xbe\xc8\xd0A\xbc\xa7\xd0A\xba\x86\xd0AQl\xd0A\xe9Q\xd0A\xb5D\xd0A\x817\xd0AM*\xd0A\xe4\x0f\xd0A\xe2\xee\xcfA\xae\xe1\xcfA\xdf\xcd\xcfA\xab\xc0\xcfA\xab\xc0\xcfA\xab\xc0\xcfA\xab\xc0\xcfAw\xb3\xcfA\xa9\x9f\xcfA\x0f\x99\xcfAu\x92\xcfA\xdb\x8b\xcfAu\x92\xcfA\xdb\x8b\xcfA\x0cx\xcfA\xd8j\xcfA\xa4]\xcfA\nW\xcfA\nW\xcfA\xa4]\xcfA\xa4]\xcfA\nW\xcfA\xd6I\xcfApP\xcfA\nW\xcfA\nW\xcfA\xa4]\xcfA\xa4]\xcfA>d\xcfA>d\xcfA\xa4]\xcfA>d\xcfA\xd8j\xcfArq\xcfA\x0cx\xcfA\xdb\x8b\xcfA\x0f\x99\xcfA\xdd\xac\xcfA\xab\xc0\xcfAE\xc7\xcfAz\xd4\xcfA\x14\xdb\xcfA\xe2\xee\xcfA\xb0\x02\xd0A\x18\x1d\xd0A\x1b>\xd0A\xb7e\xd0AT\x8d\xd0A"\xa1\xd0A\x8a\xbb\xd0A\xbe\xc8\xd0A\x8d\xdc\xd0A\xc1\xe9\xd0A\x8f\xfd\xd0A)\x04\xd1A\xc3\n\xd1A\xc3\n\xd1A]\x11\xd1A\x91\x1e\xd1A\xc6+\xd1A.F\xd1A0g\xd1A3\x88\xd1A\x01\x9c\xd1A5\xa9\xd1A\x03\xbd\xd1A8\xca\xd1A\x06\xde\xd1A\xd4\xf1\xd1A\xa2\x05\xd2A\xa0\xe4\xd1Adt\xd1A$\xc2\xd0Az\xd4\xcfA\x98\xb8\xceAN\x82\xcdA\xd1>\xccAP\xda\xcaA\x9e\x89\xc9A\xec8\xc8Al\xd4\xc6A\x88\x97\xc5A\xa4Z\xc4A\xf2\t\xc3Au\xc6\xc1A]|\xc0AGS\xbfA1*\xbeA\x1c\x01\xbdA6\xa3\xbbA\xb8_\xbaAn)\xb9AY\x00\xb8A\x0f\xca\xb6A+\x8d\xb5A'
hex = b'\x99\xfd\xb0@\xcc\xc8\xc3@\x99\xa5\xda@3\xd3\xea@39\tAf\xfa$A\x99\x95=A\x00 UA\xcc,fA\xcc\x1e\x7fAf\x08\x8cA3S\x9dAf\xa2\xafA3\x1c\xbfA\xcc\x96\xd1A\xcce\xdbA\x00u\xf5A\xe6\xf3\x00Bf\xdc\x01B\x80j\x04B\xff\xbf\xffA\x99\x03\xd7A\x00u\x98A\x99g;A\xcc\xac\xb7@\x99Y\xb5?\xcc\xcc\xb3\xbd\xcc\xcc\x94\xbc33\x82=\x99\x19\xdc>\xcc\x0c,?f\xa6\x9f>\xcc\xcc\xf1=ff\xc6=\x00\x00\x9b=\x00\x00:=33\xdf<33!=\x99\x99-=\xcc\xccR=\xcc\xcc\x94=\xcc\xcc\xb3=\xcc\xcc\xd2=\xcc\xcc\xf1=\x99\x99\xeb=ff\xe5=\x00\x00\xd9=33\xdf=\xcc\xcc\xf1=\xcc\xcc\xf1=33\xfe=\x00\x80\x0b>\xcdL\x05>\xcc\xcc\xf1=\x99\x99\xeb=\x00\x00\xd9=\xcc\xcc\xd2=\x99\x99\xcc=\x00\x00\xd9=33\xdf=33\xfe=\xccL$>\x99\x19=>\x00\x80I>3\xb3O>\x00\x00Y>\x00\x80I>3\xb30>f\xe6\x17>33\xfe=\xcc\xcc\xd2=ff\xe5=f\xe6\x17>ffF>\x99\x99k>\xccL\xe2>\xcc,G?3\xf3\x89?f&\xaf?3\xb3\x11@fF\x8c@\x99\xd5\xd4@\x00\xce\tA3\xd5$A\x00|:A\x99\xb9HA3]RA\xcc\xdaXAf\xa6]Af\x02bAf^fA3\x8dkA\x99\xedpAf\x1cvA\xcc\xc2zA\x99?~A\x00=\x80A3;\x81A3q\x82A\xcd\\\x83Af)\x84A\x00\xf6\x84A3\x97\x85A\x99>\x86A\xcd\xfe\x86Af\xcb\x87A\xcd\xb0\x88A\xcd\xa8\x89A\x99\xb9\x8aA\x00\xdd\x8bAf\x00\x8dA\xcd#\x8eA3G\x8fA\x99j\x90A3\x94\x91Af\xd0\x92A\xcd\x12\x94A3t\x95A3\x07\x97A\x00\x94\x98A3\x0e\x9aA\x00\x9b\x9bA\x00M\x9dA\xcc\xf8\x9eAfA\xa0A\x00\xa9\xa1A\x99\x8c\xa3A\xccD\xa5A\x99\xd1\xa6A\x00\xf5\xa7A3\xf3\xa8A\xcc;\xaaAfe\xabA\x99\x82\xacA\x00\x87\xadA\xccx\xaeA\x99j\xafA3V\xb0A\x99Z\xb1A3e\xb2A\xcco\xb3A\x00\x8d\xb4Af\xcf\xb5A3=\xb7A\x00\x8c\xb8A\xcc\xda\xb9A\x99)\xbbAfY\xbcA\x99W\xbdA\x99O\xbeA3;\xbfA\xccd\xc0A\x00\xc0\xc1A\x99F\xc3A\x99\xf8\xc4Af\xa4\xc6A3\x12\xc8Af\x10\xc9A\xcc\xb7\xc9A\x99-\xcaA\xccR\xcaA\xccR\xcaAf\'\xcaA\x00\x1b\xcaA3@\xcaA\xcc\x90\xcaA38\xcbA\x99\x1d\xccAf\x0f\xcdA3 \xceA\x99b\xcfA\xcc\xdc\xd0A3|\xd2A\x00f\xd4A35\xd7A\x99\x8e\xd9A\x00\x8b\xdbA\x99O\xddA\xcc\xaa\xdeA\x99\x9c\xdfA\x99\xda\xdfAf\x96\xdfA3\x14\xdfA\xccl\xdeA3\xde\xddAf\xe4\xddA\x99\xc3\xdeA\xcc{\xe0A\x00S\xe2A\x00b\xe4A\xccj\xe6Af/\xe8A\x002\xeaA\x99S\xecA\x99\x05\xeeA\xccA\xefA\xcc\xbd\xefAf\xb1\xefA\x00H\xefAf\xb9\xeeAf\xb9\xeeA\x00g\xefA\x00\x9d\xf0A\xccH\xf2A\xccW\xf4A\x99\xfb\xf6A\xff\xd0\xf9A\x99O\xfcA\xccd\xfeA\xcd\x17\x00B\xcd\xb2\x00B\x00\x16\x01B3Z\x01BM|\x01B\xb3\x04\x02BMt\x02B\xcd\xe0\x02B\x99u\x03B\xb32\x04B3\x1b\x05B3\x13\x06B\x19\xca\x06B3h\x07B\x00\xde\x07Bf(\x08B\xb3o\x08B\xcd\x91\x08B\x99\x8b\x08B\x80\xa7\x08B\x00\xf5\x08B\x00R\tBf\xbb\tBM4\nB\x19\xaa\nB3)\x0bB\x80\xae\x0bB3!\x0cB\x19{\x0cB\x80\xc5\x0cB\x19\xf7\x0cB\x80"\rB\x19T\rBf|\rB\x19\xb1\rB\x19\x0e\x0eB3\x8d\x0eBf\xf0\x0eB\x19D\x0fBM\xa7\x0fBM#\x10B\x00\x96\x10B\x00\x12\x11B\xe6\xa9\x11B\x99;\x12B\x00\xc4\x12BM*\x13B\x00~\x13B\x00\xdb\x13B\x00\x19\x14B\xcdo\x14B\xe6\xee\x14BLw\x15Bf\xf6\x15B\xe6\x81\x16BL\n\x17B\x00^\x17BL\xa5\x17B\xe6\xf5\x17B\xccO\x18B\x19\xb6\x18B\x19\x13\x19BLv\x19B\x99\xdc\x19B\x80U\x1aB\x19\xe4\x1aB\x80l\x1bB\x19\xdc\x1bBfB\x1cB3\xb8\x1cB\x99@\x1dB\xcc\x00\x1eBf\x8f\x1eBL\x08\x1fB\x19\x9d\x1fB\xcc. B\xb3\xa7 Bf\x1a!B\x19n!B\x99\x9c!B\xcc\xe0!BfP"B\x80\xb0"BL\x07#BLd#B\x19\xbb#B\xe6\xf2#B\xe6O$B\x00\xcf$BfW%B\xcc\xdf%B\xccz&B\xccS\'BL<(B3\x12)BL\xee)B\xe6\x9b*Bf\'+B3\xbc+Bf>,B\x00\xae,B\x00\xec,B\xe6\x83-B\xe6=.B\x80\xcc.B\x99K/B\x99\xa8/B\x19\xf6/B\x99C0B3\x940B\xcc\xe40B\xb3>1B\xe6\xa11BL\x0b2B\x99\x902B\x80G3BL94B3\xd14B\xe6C5B\xb3\xd85B\x99\xae6B\x80\xa37BL\x958B3k9B\x991:B\x19\x8f8B\xe6\xcf3B\x19\xb1,B\x99}!B\x00\xad\x11B\xcc\x83\xfeA\x99\x9e\xd7A\x99\x1c\xb1A\x99\xc0\x8dA\x00\xe8[A\xccz&A3\xbf\x00Af\x1e\xcf@\x99\x91\xae@3c\x9b@3\x8b\x96@3\x9f\xa3@\x00$\xc5@\xcdd\x02A\xcc\x94:A\xcdd\x82Af\xd9\xacAfq\xd8A3\xce\xfeA\xe6E\x0eBf\xfd\x18B\x99\xcb\x1fB\x00\xb8#Bf\xf2%B3"\'B\xb3\n(B3\xd4(B\x99{)B3\n*B\xe6]*B\xcc;*BLr)B\xe6m(B\x00y\'B\xccz&B3\x8f%B\xcc\x8a$B\x80,#B\xe6\n!B\xb3y\x1eB\x19\xdc\x1bB\x00m\x19BfK\x17B\x80?\x15B\x80O\x13BM\x97\x11Bf\x83\x10B\x00\x1a\x10B\xe6\x16\x10B\xe6\x92\x10B3\x94\x11Bf\xb1\x12B\x00\x9d\x13B\x99\xce\x13B\xb3t\x13Bf\xd0\x12B\xb3\x1f\x12B\x19S\x11B\x19<\x10B\x80\xb5\x0eB\x19\x16\rBMK\x0bB\xb3)\tB\xe6 \x07Bf~\x05B3\x80\x04B\x00\xdf\x03B\xcd=\x03B\xb3a\x02Bf\x7f\x01B\x99\xac\x00B\xff\x81\xffA\x99f\xfdA\xff\x06\xfbA3\x82\xf8A\xcc\xac\xf5A\x00G\xf3A\x00\xfa\xf0A\xcc\xa6\xeeA3\xc3\xecA\xcc\xe5\xeaA\xcc\x14\xe9A\x00\x88\xe7Aff\xe5Af\x9d\xe2A3\xce\xdfA\xccU\xddA34\xdbA\x99o\xd9A\x00e\xd8A\x00m\xd7A\x99Q\xd5A\x00\x8d\xd3A\xcc\x12\xd2A3\xab\xd0A\xcch\xcfA\xcc\x13\xceA\xcc\xbe\xccA\x99\x06\xcbA3P\xc8AfO\xc5A\x00<\xc2AfA\xbfA3\xcf\xbcA\x99\xcc\xbaA\x99X\xb9Af\xde\xb7Af\xcf\xb5A\x00\xb4\xb3A3\xe9\xb1A\x00o\xb0A\x99j\xafAf\xc9\xaeA\x00\x7f\xaeA\xcc\xfc\xadA\x99\xc0\xacA3@\xabA\x99\xb9\xa9A\x003\xa8A\xcc\xd7\xa6A\x99\xba\xa5A\x99\xc2\xa4Af\xc4\xa3Afo\xa2A\x00T\xa0A3K\x9eA\xcc\x8c\x9cA\x00\xe1\x9aAf\x98\x99A\x00\xd2\x98A\xcc\x11\x98A\xcc\xdb\x96Af\xc0\x94A\xcd\xdc\x92A\xcdI\x91A3\xc3\x8fA\x00h\x8eA3\x19\x8dAf/\x8bA\x99l\x88A\x99\xc2\x85A\x00D\x83A\xcd.\x81Af\x96~A\xff\xa9zA3\xb1vA\x00nrA\x00\xd4mA3_iA\x00\x1ceA\x99\xf1`A\xcc\xf8\\A3\xe7XA\xcc\xbcTA\x00\x86PA3\x11LA\xcc\xe6GAf\xfaCA\xccd@A\x00&=A\x00>:A\xccn7A\xcc\x025A\xcc\x1a2A3\x85.A3!+A\x99\xc9\'A\xccD%A\x00z#A\xcc\xe0!A\xcc\xaa A\xcc\xf0\x1fAf\xae\x1eA\x00t\x1cA\x00\xca\x19A\x00\xa4\x16A\x00\xbc\x13A\xcdh\x11A\xcd\xfc\x0eA\xcd\xd6\x0bA\x00\x98\x08A\x99\xa3\x05A\x99\xf9\x02A\xcd\xb2\x00A3\xb7\xfd@\xcc\x00\xfb@\xff\xf7\xf8@\x99\xbd\xf6@\x00(\xf3@\x00P\xee@\x00x\xe9@\xcc4\xe5@\xccT\xe1@f\xa6\xdd@3k\xd8@f>\xcb@\x99\xdd\xb4@\xcc\xb4\x97@3\x8bs@3\x836@\x00 \xf4?ff\x88?\x99\xd9\xe3>\x00\x00\xd9=\xcc\xcc\x14<\x00\x00\x00\x00\xcc\xcc\x14\xbc\x00\x00x\xbc\x00\x00x\xbcff\xc6\xbb\xcc\xcc\x14\xbc\xcc\xcc\x14\xbcff\xc6\xbb\x00\x00\x00\x00ff\xc6;ff\xc6<\x00\x00\xf8<\xcc\xcc\x14=\x99\x99-='



def _byte_to_array(data: bytearray, factor=None):
    """
    Converts bytearray containing trace values to numpy array(s). PF6 will return two arrays, min and max array
    Args:
        data: Bytes
        factor: Specific quantity factor to convert integers to floats. Only used for PF6 traces.
    Returns:
        Numpy array(s) with the extracted data
    """
    # A bug form the DDS Platform causing some traces to be compressed several times
    while data[:8] == GZIP_HEADER:
        data = gzip.decompress(data)
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

val = _byte_to_array(hex)
val2= _byte_to_array(hex2)
print(val2)
plt.plot( val2, val)
plt.show()