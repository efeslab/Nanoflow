cd ./utils
./generate_fix_trace.sh

rates=$(seq 0 50)
for rate in $rates; do
  echo "Generating traces for rate: $rate"
  bash ./generate_real_trace.sh $rate &
done

wait

echo "All traces generated"
