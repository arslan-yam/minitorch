# minitorch

# Module 1: Scalars training

## Simple dataset

6 hidden layers, lr = 1, 600 epochs

Training log

```text
Epoch: 0/600, loss: 0, correct: 0
Epoch: 10/600, loss: 44.5152, correct: 58
Epoch: 20/600, loss: 25.2582, correct: 72
Epoch: 30/600, loss: 12.7494, correct: 74
Epoch: 40/600, loss: 5.7556, correct: 79
...
Epoch: 580/600, loss: 0.0627, correct: 79
Epoch: 590/600, loss: 0.0612, correct: 79
Epoch: 600/600, loss: 0.0597, correct: 79
```

## Diag dataset

6 hidden layers, lr = 1, 600 epochs

Training log

```text
Epoch: 0/600, loss: 0, correct: 0
Epoch: 10/600, loss: 21.77272320840791, correct: 71
Epoch: 20/600, loss: 12.899259215874347, correct: 71
Epoch: 30/600, loss: 6.417284605478749, correct: 77
Epoch: 40/600, loss: 4.067989694516916, correct: 79
Epoch: 50/600, loss: 3.0278096466081257, correct: 79
Epoch: 60/600, loss: 2.4628584687405035, correct: 79
Epoch: 70/600, loss: 2.1685030464045374, correct: 79
Epoch: 80/600, loss: 1.7275689991812015, correct: 79
Epoch: 90/600, loss: 1.3000884702648001, correct: 79
Epoch: 100/600, loss: 1.0972604744362329, correct: 79
Epoch: 110/600, loss: 0.9579629848592266, correct: 79
Epoch: 120/600, loss: 0.8453303407084044, correct: 79
Epoch: 130/600, loss: 0.7524180566169135, correct: 79
Epoch: 140/600, loss: 0.6743169388916229, correct: 79
Epoch: 150/600, loss: 0.6060597355203523, correct: 79
Epoch: 160/600, loss: 0.5486473721172672, correct: 79
Epoch: 170/600, loss: 0.5006049830793154, correct: 79
Epoch: 180/600, loss: 0.45992357966714165, correct: 79
Epoch: 190/600, loss: 0.4244293145466093, correct: 79
Epoch: 200/600, loss: 0.39321434387876325, correct: 79
Epoch: 210/600, loss: 0.36559555805300603, correct: 79
Epoch: 220/600, loss: 0.3409915841923463, correct: 79
Epoch: 230/600, loss: 0.31898142989834355, correct: 79
Epoch: 240/600, loss: 0.29924939963965497, correct: 79
Epoch: 250/600, loss: 0.2814076811706983, correct: 79
Epoch: 260/600, loss: 0.2653021804840337, correct: 79
Epoch: 270/600, loss: 0.25063979995549124, correct: 79
Epoch: 280/600, loss: 0.2372917916814244, correct: 79
Epoch: 290/600, loss: 0.22508360521405837, correct: 79
Epoch: 300/600, loss: 0.2138776341345979, correct: 79
Epoch: 310/600, loss: 0.20357455712373948, correct: 79
Epoch: 320/600, loss: 0.19410392940155688, correct: 79
Epoch: 330/600, loss: 0.18539922691025154, correct: 79
Epoch: 340/600, loss: 0.17734653864315605, correct: 79
Epoch: 350/600, loss: 0.16985553456252733, correct: 79
Epoch: 360/600, loss: 0.1628800150917356, correct: 79
Epoch: 370/600, loss: 0.15636829347857734, correct: 79
Epoch: 380/600, loss: 0.15029088866238227, correct: 79
Epoch: 390/600, loss: 0.14459054236828528, correct: 79
Epoch: 400/600, loss: 0.13924596040248005, correct: 79
Epoch: 410/600, loss: 0.13423098540078368, correct: 79
Epoch: 420/600, loss: 0.129513361294033, correct: 79
Epoch: 430/600, loss: 0.12506559038285847, correct: 79
Epoch: 440/600, loss: 0.12087026063182502, correct: 79
Epoch: 450/600, loss: 0.11690785079048416, correct: 79
Epoch: 460/600, loss: 0.1131607317091114, correct: 79
Epoch: 470/600, loss: 0.10961018431674246, correct: 79
Epoch: 480/600, loss: 0.10624527624688383, correct: 79
Epoch: 490/600, loss: 0.10305437014898232, correct: 79
Epoch: 500/600, loss: 0.10002318521361417, correct: 79
Epoch: 510/600, loss: 0.09714944967504448, correct: 79
Epoch: 520/600, loss: 0.09441981056950233, correct: 79
Epoch: 530/600, loss: 0.09181841978100706, correct: 79
Epoch: 540/600, loss: 0.08933471715402802, correct: 79
Epoch: 550/600, loss: 0.08696473553487104, correct: 79
Epoch: 560/600, loss: 0.0847012583646097, correct: 79
Epoch: 570/600, loss: 0.08253670016017665, correct: 79
Epoch: 580/600, loss: 0.08046518256484689, correct: 79
Epoch: 590/600, loss: 0.07847852903623724, correct: 79
Epoch: 600/600, loss: 0.07657684199587834, correct: 79
```

## Split dataset

6 hidden layers, lr = 0.5, 600 epochs

Training log

```text
Epoch: 0/600, loss: 0, correct: 0
Epoch: 10/600, loss: 53.08052480470115, correct: 49
Epoch: 20/600, loss: 52.94823296172844, correct: 49
Epoch: 30/600, loss: 52.92055607854274, correct: 49
Epoch: 40/600, loss: 52.891489865251216, correct: 49
Epoch: 50/600, loss: 52.8552701193039, correct: 49
Epoch: 60/600, loss: 52.81254009655919, correct: 49
Epoch: 70/600, loss: 52.75513967953077, correct: 50
Epoch: 80/600, loss: 52.644740023248765, correct: 51
Epoch: 90/600, loss: 52.50146537142221, correct: 51
Epoch: 100/600, loss: 52.30531129213718, correct: 51
Epoch: 110/600, loss: 52.03577705762164, correct: 50
Epoch: 120/600, loss: 51.6439178649552, correct: 50
Epoch: 130/600, loss: 51.10371145112955, correct: 50
Epoch: 140/600, loss: 50.37203519261439, correct: 50
Epoch: 150/600, loss: 49.3131368611039, correct: 50
Epoch: 160/600, loss: 47.80536922449101, correct: 50
Epoch: 170/600, loss: 45.63854364737416, correct: 52
Epoch: 180/600, loss: 41.835621339805655, correct: 59
Epoch: 190/600, loss: 37.49193211810715, correct: 70
Epoch: 200/600, loss: 33.25761991579542, correct: 69
Epoch: 210/600, loss: 40.00586934062798, correct: 56
Epoch: 220/600, loss: 34.94890722432566, correct: 61
Epoch: 230/600, loss: 33.02554115068692, correct: 61
Epoch: 240/600, loss: 32.59003040215845, correct: 61
Epoch: 250/600, loss: 31.58982996316748, correct: 61
Epoch: 260/600, loss: 30.881649049884828, correct: 62
Epoch: 270/600, loss: 30.043964994418424, correct: 62
Epoch: 280/600, loss: 29.665323579023536, correct: 62
Epoch: 290/600, loss: 29.344058054148228, correct: 63
Epoch: 300/600, loss: 29.1299765709555, correct: 63
Epoch: 310/600, loss: 28.640849959658357, correct: 63
Epoch: 320/600, loss: 28.286690388668593, correct: 63
Epoch: 330/600, loss: 28.089161196687684, correct: 63
Epoch: 340/600, loss: 28.027598637190284, correct: 63
Epoch: 350/600, loss: 27.577968223665057, correct: 64
Epoch: 360/600, loss: 27.501691261382692, correct: 63
Epoch: 370/600, loss: 26.715656083722482, correct: 64
Epoch: 380/600, loss: 26.841078868785456, correct: 64
Epoch: 390/600, loss: 26.759408910534898, correct: 63
Epoch: 400/600, loss: 25.442320061161297, correct: 64
Epoch: 410/600, loss: 25.43685085394669, correct: 64
Epoch: 420/600, loss: 26.359162206802196, correct: 65
Epoch: 430/600, loss: 25.882589632779524, correct: 64
Epoch: 440/600, loss: 23.38691149447911, correct: 65
Epoch: 450/600, loss: 23.156983178549257, correct: 65
Epoch: 460/600, loss: 29.146098552795454, correct: 64
Epoch: 470/600, loss: 23.67277352109039, correct: 68
Epoch: 480/600, loss: 22.827061908212844, correct: 66
Epoch: 490/600, loss: 20.352734865856895, correct: 69
Epoch: 500/600, loss: 20.87274985709891, correct: 68
Epoch: 510/600, loss: 55.386371749511646, correct: 56
Epoch: 520/600, loss: 20.03335480658628, correct: 70
Epoch: 530/600, loss: 21.911665919236746, correct: 67
Epoch: 540/600, loss: 21.461263332855093, correct: 68
Epoch: 550/600, loss: 22.995552558981306, correct: 67
Epoch: 560/600, loss: 22.809772364280107, correct: 68
Epoch: 570/600, loss: 22.061272202652646, correct: 68
Epoch: 580/600, loss: 20.689681072890277, correct: 69
Epoch: 590/600, loss: 20.99751803325342, correct: 69
Epoch: 600/600, loss: 19.327285306873495, correct: 71
```

## Xor dataset

6 hidden layers, lr = 0.5, 600 epochs

Training log

```text
Epoch: 0/600, loss: 0, correct: 0
Epoch: 10/600, loss: 53.53514848791409, correct: 48
Epoch: 20/600, loss: 51.8888656418426, correct: 52
Epoch: 30/600, loss: 50.15381879350942, correct: 59
Epoch: 40/600, loss: 47.973205172364324, correct: 58
Epoch: 50/600, loss: 47.81732227609509, correct: 47
Epoch: 60/600, loss: 45.22864523190257, correct: 53
Epoch: 70/600, loss: 44.27179585220543, correct: 58
Epoch: 80/600, loss: 41.968962469659644, correct: 63
Epoch: 90/600, loss: 39.996902691625195, correct: 68
Epoch: 100/600, loss: 36.83977066629235, correct: 72
Epoch: 110/600, loss: 36.03166592133366, correct: 67
Epoch: 120/600, loss: 32.435759698787386, correct: 69
Epoch: 130/600, loss: 29.143545522947818, correct: 69
Epoch: 140/600, loss: 23.418701781453198, correct: 73
Epoch: 150/600, loss: 22.42212934479872, correct: 73
Epoch: 160/600, loss: 20.784922841318846, correct: 73
Epoch: 170/600, loss: 18.505920997116444, correct: 74
Epoch: 180/600, loss: 16.20691807666616, correct: 77
Epoch: 190/600, loss: 15.113076272070296, correct: 78
Epoch: 200/600, loss: 14.935910285894213, correct: 77
Epoch: 210/600, loss: 16.354256912290037, correct: 74
Epoch: 220/600, loss: 13.591264511993536, correct: 77
Epoch: 230/600, loss: 12.34397028882649, correct: 78
Epoch: 240/600, loss: 13.3029731584646, correct: 76
Epoch: 250/600, loss: 11.506288064241177, correct: 77
Epoch: 260/600, loss: 11.252787402176237, correct: 77
Epoch: 270/600, loss: 10.677080968324768, correct: 77
Epoch: 280/600, loss: 10.663587057854885, correct: 77
Epoch: 290/600, loss: 10.878367768957368, correct: 77
Epoch: 300/600, loss: 9.466366434775624, correct: 77
Epoch: 310/600, loss: 9.163814036801712, correct: 77
Epoch: 320/600, loss: 8.96919347880294, correct: 77
Epoch: 330/600, loss: 8.07080295577009, correct: 79
Epoch: 340/600, loss: 8.333351123815648, correct: 78
Epoch: 350/600, loss: 7.758050644304311, correct: 79
Epoch: 360/600, loss: 7.688461785696689, correct: 78
Epoch: 370/600, loss: 7.618584561050116, correct: 77
Epoch: 380/600, loss: 7.828786035134807, correct: 77
Epoch: 390/600, loss: 6.854225008627354, correct: 79
Epoch: 400/600, loss: 6.959818691296855, correct: 79
Epoch: 410/600, loss: 6.332715814513706, correct: 79
Epoch: 420/600, loss: 6.296237993121722, correct: 79
Epoch: 430/600, loss: 6.133971523726673, correct: 79
Epoch: 440/600, loss: 6.972330301040306, correct: 79
Epoch: 450/600, loss: 6.0702294006696595, correct: 79
Epoch: 460/600, loss: 6.569432937051031, correct: 77
Epoch: 470/600, loss: 8.812388021558505, correct: 76
Epoch: 480/600, loss: 5.2769386643837874, correct: 79
Epoch: 490/600, loss: 4.959714526682311, correct: 79
Epoch: 500/600, loss: 5.305359480618012, correct: 79
Epoch: 510/600, loss: 6.963202608845734, correct: 79
Epoch: 520/600, loss: 5.4398589469684255, correct: 79
Epoch: 530/600, loss: 6.923347236407303, correct: 79
Epoch: 540/600, loss: 4.624498892128703, correct: 79
Epoch: 550/600, loss: 3.948736924325765, correct: 79
Epoch: 560/600, loss: 3.7316334368339508, correct: 79
Epoch: 570/600, loss: 3.496513977757089, correct: 79
Epoch: 580/600, loss: 3.3694396984445225, correct: 78
Epoch: 590/600, loss: 3.2996182885482415, correct: 78
Epoch: 600/600, loss: 3.2438142592341936, correct: 78
```

---

# Module 2: Tensor Training

50 points, 5 hidden layers, lr = 0.5, 500 epochs, time/epoch = 0.046s

## Simple dataset

Training log

```
Epoch  10  loss  30.407 correct 30
Epoch  50  loss  7.205 correct 48
Epoch  100  loss  6.595 correct 48
Epoch  200  loss  5.413 correct 48
Epoch  300  loss  4.799 correct 48
Epoch  400  loss  4.305 correct 48
Epoch  500  loss  3.695 correct 48
  time/epoch: 0.046s
```

## Diag dataset

Training log

```
Epoch  10  loss  22.631 correct 41
Epoch  50  loss  17.162 correct 41
Epoch  100  loss  5.300 correct 49
Epoch  200  loss  3.863 correct 47
Epoch  300  loss  3.106 correct 48
Epoch  400  loss  2.672 correct 48
Epoch  500  loss  2.574 correct 48
  time/epoch: 0.046s
```

## Split dataset

Training log

```
Epoch  10  loss  34.137 correct 30
Epoch  50  loss  32.628 correct 34
Epoch  100  loss  23.678 correct 43
Epoch  200  loss  7.951 correct 46
Epoch  300  loss  5.843 correct 49
Epoch  400  loss  3.946 correct 49
Epoch  500  loss  4.846 correct 48
  time/epoch: 0.046s
```

## Xor dataset

Training log

```
Epoch  10  loss  33.979 correct 19
Epoch  50  loss  30.340 correct 34
Epoch  100  loss  24.681 correct 35
Epoch  200  loss  14.954 correct 45
Epoch  300  loss  10.582 correct 47
Epoch  400  loss  8.080 correct 48
Epoch  500  loss  7.715 correct 48
  time/epoch: 0.046s
```

---

# Module 3: Efficiency improvements

50 points, 100 epochs

## Simple dataset

### CPU

Training log

```text
Epoch  0  loss  5.2378367938354025 correct 45
  time/epoch: 27.471797227859497s
Epoch  10  loss  1.9905409674358292 correct 50
  time/epoch: 0.03902745246887207s
Epoch  20  loss  0.722818691925663 correct 47
  time/epoch: 0.03612923622131348s
Epoch  30  loss  1.0443822858111493 correct 50
  time/epoch: 0.038548946380615234s
Epoch  40  loss  0.7241019724698469 correct 50
  time/epoch: 0.037558555603027344s
Epoch  50  loss  0.37141934423989464 correct 49
  time/epoch: 0.035471200942993164s
Epoch  60  loss  0.13802424549369116 correct 50
  time/epoch: 0.0354611873626709s
Epoch  70  loss  0.47752179562409536 correct 50
  time/epoch: 0.03502082824707031s
Epoch  80  loss  0.674895118218075 correct 49
  time/epoch: 0.035843610763549805s
Epoch  90  loss  1.071454385851866 correct 50
  time/epoch: 0.04048871994018555s
```

### GPU

Training log

```text
Epoch  0  loss  4.812581733379714 correct 43
  time/epoch: 3.088108539581299s
Epoch  10  loss  2.389976357072972 correct 48
  time/epoch: 0.779613733291626s
Epoch  20  loss  0.9223367229193951 correct 50
  time/epoch: 0.7920539379119873s
Epoch  30  loss  0.6132368148252241 correct 50
  time/epoch: 0.8692193031311035s
Epoch  40  loss  0.8443921180747713 correct 50
  time/epoch: 0.8171367645263672s
Epoch  50  loss  0.539672228722854 correct 50
  time/epoch: 0.8156919479370117s
Epoch  60  loss  0.15151644235204018 correct 50
  time/epoch: 0.7995297908782959s
Epoch  70  loss  0.056512087144279514 correct 50
  time/epoch: 0.8019511699676514s
Epoch  80  loss  0.057348705259877446 correct 50
  time/epoch: 0.8916285037994385s
Epoch  90  loss  0.023220764502509192 correct 50
  time/epoch: 0.7820770740509033s
```

## Diag dataset

### CPU

Training log

```text
Epoch  0  loss  2.4503356163499106 correct 45
  time/epoch: 12.99582028388977s
Epoch  10  loss  0.3671714983000165 correct 49
  time/epoch: 0.035866737365722656s
Epoch  20  loss  0.20580124582483678 correct 49
  time/epoch: 0.03166317939758301s
Epoch  30  loss  0.25237600190572596 correct 50
  time/epoch: 0.03523445129394531s
Epoch  40  loss  0.6408666372827683 correct 49
  time/epoch: 0.03250241279602051s
Epoch  50  loss  0.8882572223671452 correct 50
  time/epoch: 0.033388614654541016s
Epoch  60  loss  0.27522472761321054 correct 50
  time/epoch: 0.03245830535888672s
Epoch  70  loss  0.008294527807608074 correct 50
  time/epoch: 0.03218579292297363s
Epoch  80  loss  0.39712606631161157 correct 50
  time/epoch: 0.03325295448303223s
Epoch  90  loss  0.04743764476427284 correct 50
  time/epoch: 0.03372812271118164s
```

### GPU

Training log

```text
Epoch  0  loss  3.7089225770856675 correct 41
  time/epoch: 3.005887508392334s
Epoch  10  loss  1.2949402216750312 correct 46
  time/epoch: 0.8675622940063477s
Epoch  20  loss  1.700008636550403 correct 49
  time/epoch: 0.8622338771820068s
Epoch  30  loss  1.1153664159054668 correct 49
  time/epoch: 0.9243922233581543s
Epoch  40  loss  1.8908194807473857 correct 49
  time/epoch: 0.8204624652862549s
Epoch  50  loss  0.903663283152395 correct 49
  time/epoch: 0.8425130844116211s
Epoch  60  loss  0.5759804551869384 correct 49
  time/epoch: 0.8052170276641846s
Epoch  70  loss  0.5469167552666898 correct 50
  time/epoch: 0.810882568359375s
Epoch  80  loss  1.1383499239403978 correct 50
  time/epoch: 0.9820647239685059s
Epoch  90  loss  0.9168228366295759 correct 50
  time/epoch: 0.8681612014770508s
```

## Split dataset

### CPU

Training log

```text
Epoch  0  loss  8.585140801859719 correct 29
  time/epoch: 27.868874073028564s
Epoch  10  loss  5.829099953145586 correct 36
  time/epoch: 0.033423662185668945s
Epoch  20  loss  5.131371755476135 correct 43
  time/epoch: 0.03310799598693848s
Epoch  30  loss  3.7912860330442473 correct 43
  time/epoch: 0.03279924392700195s
Epoch  40  loss  3.062014013845069 correct 45
  time/epoch: 0.04085969924926758s
Epoch  50  loss  4.6817390185586545 correct 46
  time/epoch: 0.03735065460205078s
Epoch  60  loss  2.7666305564720233 correct 48
  time/epoch: 0.03743100166320801s
Epoch  70  loss  1.572545257994916 correct 48
  time/epoch: 0.034235239028930664s
Epoch  80  loss  0.6889887890964899 correct 48
  time/epoch: 0.03398728370666504s
Epoch  90  loss  2.861258392248888 correct 48
  time/epoch: 0.033847808837890625s
```

### GPU

Training log

```text
Epoch  0  loss  4.9268867608509055 correct 31
  time/epoch: 3.2422914505004883s
Epoch  10  loss  9.377686772383592 correct 32
  time/epoch: 0.9014341831207275s
Epoch  20  loss  5.553370451407149 correct 41
  time/epoch: 0.9138057231903076s
Epoch  30  loss  4.367019065017009 correct 40
  time/epoch: 0.9994316101074219s
Epoch  40  loss  4.232833231771651 correct 48
  time/epoch: 0.857952356338501s
Epoch  50  loss  3.0817162287551536 correct 43
  time/epoch: 0.8517899513244629s
Epoch  60  loss  3.6305963288925187 correct 47
  time/epoch: 0.8609640598297119s
Epoch  70  loss  2.168981859256876 correct 48
  time/epoch: 0.8998012542724609s
Epoch  80  loss  1.9577700413579848 correct 50
  time/epoch: 0.9080259799957275s
Epoch  90  loss  2.1490025820297163 correct 48
  time/epoch: 0.8182260990142822s
```
