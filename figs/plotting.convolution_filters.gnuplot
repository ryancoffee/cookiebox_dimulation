#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.2 patchlevel 2    last modified 2017-11-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2017
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
unset grid
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
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels auto 5
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
set ytics  norangelimit logscale autofreq 
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
set xlabel "" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 0.00000 : 200.000 ] noreverse nowriteback
set x2range [ -247.231 : 671.055 ] noreverse nowriteback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ 4.88281e-05 : 1.00000 ] noreverse nowriteback
set y2range [ -0.0733657 : 0.0417215 ] noreverse nowriteback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse nowriteback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse nowriteback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse nowriteback
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
file(i) = sprintf('/media/coffee/scratch/hamamatsu/dec2018/raw/CookieBox_waveforms.4pulses.image%04i.dat',i)
analog(i) = sprintf('/media/coffee/scratch/hamamatsu/dec2018/processed/CookieBox_waveforms.4pulses.image%04i.analogprocess.out',i)
theory(i) = sprintf('/media/coffee/scratch/hamamatsu/dec2018/processed/CookieBox_waveforms.4pulses.image%04i.analogtheory.out',i)
GNUTERM = "qt"
GPFUN_file = "file(i) = sprintf('/media/coffee/scratch/hamamatsu/dec2018/raw/CookieBox_waveforms.4pulses.image%04i.dat',i)"
GPFUN_analog = "analog(i) = sprintf('/media/coffee/scratch/hamamatsu/dec2018/processed/CookieBox_waveforms.4pulses.image%04i.analogprocess.out',i)"
GPFUN_theory = "theory(i) = sprintf('/media/coffee/scratch/hamamatsu/dec2018/processed/CookieBox_waveforms.4pulses.image%04i.analogtheory.out',i)"
i = 4901
orig = 7
sig = 9
dsig = 10
ddsig = 11
thresh = 12
deltas = 13
sfilt = 15 
dfilt = 16 
ddfilt = 17 

## Last datafile plotted: "/media/coffee/scratch/hamamatsu/dec2018/processed/CookieBox_waveforms.4pulses.image4901.analogprocess.out"
#set term png size 1800,1200
#set output 'figs/plotting.convolution_filters.png'
set term post eps size 6in,4in
set output 'figs/plotting.convolution_filters.eps'
set multiplot
set key top left
set size .333,.333
buf = .035
set lmargin screen buf
set rmargin screen 0.333-buf
set origin 0,.666
set auto y
set xrange [2200:2700]
set xtics 2200,100,2700
llim = -.4
hlim = .2
scale = 10
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
set y2tics nomirror
set ytics nomirror
plot	theory(i) u 0:orig lc -1 title 'original signal',\
	theory(i) u 0:sig lc 1 title 'theory signal',\
	analog(i) u 0:sig lc 2 axes x1y2 title 'analog signal'
	
set origin 0,.333
llim = -.06
hlim = .06
scale = 25
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot theory(i) u 0:dsig title 'theory deriv',\
	analog(i) u 0:dsig axes x1y2 title 'analog deriv'
set origin 0,0
llim = -.01
hlim = .02
scale = 25
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot theory(i) u 0:ddsig title 'theory dderiv',\
	analog(i) u 0:ddsig axes x1y2 title 'analog dderiv'
set xrange [0:200]
set lmargin screen 0.666+buf
set rmargin screen 1.0-buf
set origin 0.666,.666
set xtics 0,50,200
llim = -.025
hlim = .1
scale = 12.5
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot theory(i) u 0:sfilt title 'theory conv',\
	analog(i) u 0:sfilt axes x1y2 title 'analog conv'
set origin 0.666,.333
llim = -.025
hlim = .025
scale = 25
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot theory(i) u 0:dfilt title 'theory dconv',\
	analog(i) u 0:dfilt axes x1y2 title 'analog dconv'
set origin 0.666,0
llim = -.01
hlim = .01
scale = 25
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot theory(i) u 0:ddfilt title 'theory ddconv',\
	analog(i) u 0:ddfilt axes x1y2 title 'analog ddconv'

set xrange [2200:2700]
set xtics 2200,100,2700
set lmargin screen 0.333+buf
set rmargin screen 0.666-buf
set origin 0.333,.5
llim = -.006
hlim = .002
scale = 250
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot (-0.0005) lw 2 lc -1 title 'thresh' ,\
	theory(i) u 0:thresh lc 1 title 'theory sig*ddsig',\
	analog(i) u 0:thresh axes x1y2 lc 2 title 'analog sig*ddsig'
set origin 0.333,.125
llim = -.06
hlim = .06
scale = 25
set yrange [llim:hlim]
set y2range [scale*llim:scale*hlim]
plot (0) lw 2 lc -1 title 'zero crossing',\
	theory(i) u 0:(($13>0?$10:0./0)) lc 1 title 'theory thresh dsig',\
	analog(i) u 0:(($13>0?$10:0./0)) axes x1y2 lc 2 title 'analog thresh dsig'
set origin 0.333,0
unset multiplot
#    EOF
