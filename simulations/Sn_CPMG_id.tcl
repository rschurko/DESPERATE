spinsys {
  channels 119Sn
  nuclei 119Sn
  shift 1 0p -658.6p 0 0 0 0
}

par {
  crystal_file     zcw4180
  num_cores        18

  sw               500000
  variable tsw     1e6/sw
  variable r       25
  np               512*r
  proton_frequency 600e6
  start_operator   I1z
  detect_operator  I1p
  method           direct
  gamma_angles     1
  spin_rate        0
  verbose          1101
  
  variable p1      5
  variable de	   10
  variable n	   (np/r)
} 

proc pulseq {} {
  global par
  
  pulseid 5 50e3 0
  delay $par(d1c)
  delay $par(de)
  pulseid 10 50e3 90
  delay $par(de)
  
  for {set i 0} {$i < $par(r)} {incr i} {
    for {set j 0} {$j < $par(n)} {incr j} {
      acq
      delay $par(tsw)
    }
    delay $par(de)
    pulseid 10 50e3 90
    delay $par(de)
  }
}

proc main {} {
  global par 
  
  set par(d1c) [expr ($par(n)*$par(tsw)/2)]

  set f [fsimpson]
  
  #Gaussian Parameters; sd is like inverse of lb, and mu should stay 1
  set par(sd) 100
  set par(mu) 1
  
  set indx1 [expr -(($par(np)/$par(r))/2)]
  set indx2 [expr (($par(np)/$par(r))/2)]
  
  for {set n 0} {$n < $par(r)} {incr n} {
	for {set i $indx1} {$i < $indx2} {incr i} {
		set par(gauss) [expr ((1/(2*3.14*$par(sd)))*exp(-(pow(($i-$par(mu)),2))/(2*pow($par(sd),2))))]
		set j  [expr ($i + $indx2 + 1 + ($n*($par(np)/$par(r))))]
		set c  [findex $f $j]
		set re [lindex $c 0]
		set im [lindex $c 1]
		fsetindex $f $j [expr $re*$par(gauss)] [expr $im*$par(gauss)]
	 }
	}
  
  fsave $f $par(name)_$par(r).fid
  fzerofill $f 131072
  fft $f
  fabs $f
  fsave $f $par(name)_abs.spe
  funload $f
}
