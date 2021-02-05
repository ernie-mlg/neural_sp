for i in $(lspci -d "10b5:" | awk '{print $1}') ; do
       o=$(lspci -vvv -s $i | grep ACSCtl)
       if [ $? -eq 0 ] ; then
               echo $o | grep "+"
               if [ $? -eq 0 ] ; then
                       setpci -s $i f2a.w=0000
               fi
       fi
done
