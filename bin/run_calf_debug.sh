for i in 1 2 3
    do
        if [ $i = "1" ]; then
            ./bin/run_debug.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} $i

            nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
            sleep 5
        else
            ./bin/train_debug.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} $i

            nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
            sleep 5

            ./bin/run_from_saved_debug.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} $i

            status=$?

            echo $status

            nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
            sleep 5

            if [ $status -ne 0 ]; then
                break
            fi
        fi
    done

./bin/val_from_saved_debug.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} $i

nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
sleep 5