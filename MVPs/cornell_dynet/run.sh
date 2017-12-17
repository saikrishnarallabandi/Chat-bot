exp_name='../exp/basic'

mkdir -p $exp_name/logs

for hidden in 128 
   do
    python 000_baseline_chat.py $exp_name $hidden
   done

