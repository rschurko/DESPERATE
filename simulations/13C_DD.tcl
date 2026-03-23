spinsys {
  channels 13C
  nuclei 13C 13C
shift 1 0.0p -300p 0.9 8 0 0
shift 2 0.0p -300p 0.9 -8 0 0
dipole 1 2 -2890 0 90 90
}

par {
  crystal_file     zcw4180
  num_cores        9
  
  sw               1e6
  variable tsw     1e6/sw
  np               1024
proton_frequency 233.3384e6
  start_operator   Inx
  detect_operator  Inp
  method           cheby1
  gamma_angles     1
  spin_rate        0
  verbose          0000
  
  #conjugate_fid    true
} 

proc pulseq {} {
  global par 
offset 0
  acq_block { delay $par(tsw) } 

}

proc main {} {
  global par 

  set f [fsimpson]
  fsave $f $par(name).fid
  funload $f
}