for session_path in $(find session -name server); do
  base_dir=$(dirname $session_path)
  if ! test -d ".real_${base_dir}"; then
    mkdir -p ".real_${base_dir}"
    cp -r ${base_dir} $(dirname ".real_${base_dir}")
  fi
  env session_path=".real_${session_path}" python3 exp_analyzer.py
done
