@echo off
title High Precision Latency Timer
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$sw = [System.Diagnostics.Stopwatch]::StartNew(); ^
 write-host 'Press Ctrl+C to stop' -ForegroundColor Cyan; ^
 while($true) { ^
    $ts = $sw.Elapsed; ^
    $time = '{0:00}:{1:00}:{2:00}.{3:000}' -f $ts.Hours, $ts.Minutes, $ts.Seconds, $ts.Milliseconds; ^
    write-host `r$time -NoNewline; ^
 }"