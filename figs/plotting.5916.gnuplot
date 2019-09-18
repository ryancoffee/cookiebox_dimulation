#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.2 patchlevel 6    last modified 2019-01-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2018
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ"
#    	immediate help:   type "help"  (plot window: hit 'h')
# set terminal qt 0 font "Sans,9"
# set output
unset clip points
set clip one
unset clip two
set errorbars front 1.000000 
set border 31 front lt black linewidth 1.000 dashtype solid
set zdata 
set ydata 
set xdata 
set y2data 
set x2data 
set boxwidth
set style fill  empty border
set style rectangle back fc  bgnd fillstyle   solid 1.00 border lt -1
set style circle radius graph 0.02 
set style ellipse size graph 0.05, 0.03 angle 0 units xy
set dummy x, y
set format x "% h" 
set format y "% h" 
set format x2 "% h" 
set format y2 "% h" 
set format z "% h" 
set format cb "% h" 
set format r "% h" 
set ttics format "% h"
set timefmt "%d/%m/%y,%H:%M"
set angles radians
set tics back
set grid nopolar
set grid xtics nomxtics ytics nomytics noztics nomztics nortics nomrtics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   lt 0 linecolor 0 linewidth 0.500,  lt 0 linecolor 0 linewidth 0.500
unset raxis
set theta counterclockwise right
set style parallel front  lt black linewidth 2.000 dashtype solid
set key title "" center
set key fixed right top vertical Right noreverse enhanced autotitle nobox
set key noinvert samplen 4 spacing 1 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
unset arrow
set style increment default
unset style line
unset style arrow
set style histogram clustered gap 2 title textcolor lt -1
unset object
set style textbox transparent margins  1.0,  1.0 border  lt -1 linewidth  1.0
set offsets 0, 0, 0, 0
set pointsize 1
set pointintervalbox 1
set encoding default
unset polar
unset parametric
unset decimalsign
unset micro
unset minussign
set view 60, 30, 1, 1
set view azimuth 0
set rgbmax 255
set samples 100, 100
set isosamples 10, 10
set surface 
unset contour
set cntrlabel  format '%8.3g' font '' start 5 interval 20
set mapping cartesian
set datafile separator ","
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels auto 5 unsorted
set cntrparam firstlinetype 0
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data lines
set style function lines
unset xzeroaxis
unset yzeroaxis
unset zzeroaxis
unset x2zeroaxis
unset y2zeroaxis
set xyplane relative 0.5
set tics scale  1, 0.5, 1, 1, 1
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set mrtics default
set nomttics
set xtics border in scale 1,0.5 mirror norotate  autojustify
set xtics  norangelimit autofreq 
set ytics border in scale 1,0.5 mirror norotate  autojustify
set ytics  norangelimit autofreq 
set ztics border in scale 1,0.5 nomirror norotate  autojustify
set ztics  norangelimit autofreq 
unset x2tics
unset y2tics
set cbtics border in scale 1,0.5 mirror norotate  autojustify
set cbtics  norangelimit autofreq 
set rtics axis in scale 1,0.5 nomirror norotate  autojustify
set rtics  norangelimit autofreq 
unset ttics
set title "" 
set title  font "" norotate
set timestamp bottom 
set timestamp "" 
set timestamp  font "" norotate
set trange [ * : * ] noreverse nowriteback
set urange [ * : * ] noreverse nowriteback
set vrange [ * : * ] noreverse nowriteback
set xlabel "ToF [ns]" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 3565.00 : 3585.00 ] noreverse writeback
set x2range [ 3649.15 : 3658.60 ] noreverse writeback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ -0.0291266 : 0.0304014 ] noreverse writeback
set y2range [ -0.0229938 : 0.0331189 ] noreverse writeback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse writeback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse writeback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse writeback
unset logscale
unset jitter
set zero 1e-08
set lmargin  -1
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "en_US.UTF-8"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles noborder corners2color mean
set pm3d nolighting
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
set style boxplot candles range  1.50 outliers pt 7 separation 1 labels auto unsorted
set loadpath 
set fontpath 
set psdir
set fit brief errorvariables nocovariancevariables errorscaling prescale nowrap v5
fewfile(i) = sprintf('few_0_200_2000_2300/C4--C4_1628--%05i.txt',i)
file5916(i) = sprintf('tens_0_1000_2400_2700_ZX60-5916/C4--C4_1628--%05i.txt',i)
file5916_lowsig(i) = sprintf('tens_0_1000_2200_2500_ZX60-5916/C4--C4_1628--%05i.txt',i)
file_lowsig(i) = sprintf('tens_0_1000_2200_2500/C4--C4_1628--%05i.txt',i)
file(i) = sprintf('tens_0_1000_2400_2700/C4--C4_1628--%05i.txt',i)
GNUTERM = "qt"
file = "C4--C4_1628--00102.txt"
file2 = "../thousands_0_1000_2400_2700_0mesh_ZX60-5916_lowpass/C4--C4_1628--01001.txt"
fewfile = "few_0_200_2000_2300/C4--C4_1628--00111.txt"
i = 200
## Last datafile plotted: "tens_0_1000_2400_2700_ZX60-5916/C4--C4_1628--00200.txt"
unset grid
set xrange [3566.75:3567.75]
set yrange [-.03:.03]
set y2range [-.12:.12]
set y2tics auto
set ytics nomirror
set term png size 1200,1200
set output 'plotting.5916.png'
set multiplot
set size 1,1
set origin 0,0
set xlabel 'ToF [ns]'
set arrow 1 from 3567.2,-.025 to 3567.1,-.025 lc 1 lw 2
set arrow 2 from 3567.6,.025 to 3567.7,.025 lc 2 lw 2 
set y2tics (-.1,-.05,-.01,0,.01 ,.05 ,.1 )
plot file(i+2) u (1e9*$1):2 every ::5 lw 2 title 'straight', file5916(i) u (1e9*$1+5.36):2 every ::5 axes x1y2 lw 2 title 'ZX60-5916',\
file5916(i) u (1e9*$1+5.36):(abs(1e9*$1+5.36 - 3567.3) < .05? $2:0./0) every ::5 axes x1y2 w impulses lw 2 notitle, \
(0) lc -1 notitle,(-0.01) axes x1y2 lc -1 notitle,(0.01) axes x1y2 lc -1 notitle
set xrange [3565:3585]
set size .5,.35
set origin .07,.57
unset xlabel
unset arrow 1
unset arrow 2
plot file(i+2) u (1e9*$1):2 every ::5 title 'straight', file5916(i) u (1e9*$1+5.36):2 every ::5 axes x1y2 title 'ZX60-5916'
unset multiplot

#    EOF
