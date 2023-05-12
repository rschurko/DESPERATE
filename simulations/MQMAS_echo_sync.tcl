spinsys {
  channels 87Rb
  nuclei 87Rb 87Rb 87Rb
  quadrupole 1 2 1.68e6 0.2 0 0 0
  quadrupole 2 2 1.94e6 1.0 0 0 0
  quadrupole 3 2 1.72e6 0.5 0 0 0
  shift 1 -27.4p 0 0 0 0 0
  shift 2 -28.5p 0 0 0 0 0
  shift 3 -31.3p 0 0 0 0 0
}

par {
  num_cores        18
  proton_frequency 600e6
  method           taylor
  gamma_angles     5
  spin_rate        10e3
  crystal_file     zcw28656
  start_operator   Inz
  detect_operator  Inc
  sw               10e3
  verbose          1101
  variable tsw     1e6/sw
  
  variable np2    1024
  variable np1    128
  
  np 			  np2*np1
  
  
}

proc pulseq {} {
  global par
  
  ##Regular t1 and echo, 3 pulse 
  
  matrix set 1 totalcoherence {3}
  matrix set 2 totalcoherence {1}
  matrix set 3 totalcoherence {-1}


  for {set i 0} {$i < $par(np1)} {incr i} {
  
  offset -6100
  
  pulseid 7 100e3 y  
  filter 1
  
  #delay [expr 100*$i*0.5625]
  delay [expr 100*$i]
  
  pulseid 2 100e3 -y
  filter 2
  
  #delay [expr 100*$i*0.4375]
  delay $par(tau)
  
  pulseid 50 10e3 y
  filter 3
  
  delay $par(de)
  
  for {set j 1} {$j <= $par(np2)} {incr j} {
  acq
  delay $par(tsw)
 }
 
 reset
}
}

proc main {} {
  global par

  set par(d6) [expr ($par(np2)*($par(tsw)))]

  #rotor cycle amount and sync, note ideal pulses so no pulse widths
  set par(M)  518
  
  set par(de) [expr ((2*$par(M))*(1e6/$par(spin_rate))-$par(d6))/2]
  
  set par(tau) [expr $par(d6)/2+$par(de)]
  
  set f [fsimpson]
  fsave $f $par(name).fid
}
