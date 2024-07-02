#! /bin/bash
IFS=","
IN=$(tail -1 $1)
#echo $IN

#echo "set askcc=False;" >> ~/.mailrc
ttycharset=UTF-8

read -ra array <<< "$IN"

if awk "BEGIN {exit !(${array[1]} > $3)}";then
	echo -e "BLTM: Last entry of logfile: $1 is above limit given as: [$2 - $3] °C, actual value: ${array[1]} °C"
	echo "Last entry of logfile: $1 is above limits given as: [$2 - $3] °C, actual value: ${array[1]} °C" | mailx -s "BLTM: Temperature sensor reading exceeds limit." -a "From: Anders Celsius <temp-monitor@batterilabbet.elteknik.chalmers.se>" -a "Content-Type: text/html; charset=UTF-8" jutsell@chalmers.se 

fi

if awk "BEGIN {exit !(${array[1]} < $2)}";then
	echo "BLTM: Last entry of logfile: $1 is below limit given as: [$2 - $3] °C, actual value: ${array[1]} °C"
	echo "Last entry of logfile: $1 is below limits given as: [$2 - $3] °C, actual value: ${array[1]} °C" | mailx -s "BLTM: Temperature sensor reading exceeds limit." -a "From: Anders Celsius <temp-monitor@batterilabbet.elteknik.chalmers.se>" -a "Content-Type: text/html; charset=UTF-8" jutsell@chalmers.se 
	
fi


