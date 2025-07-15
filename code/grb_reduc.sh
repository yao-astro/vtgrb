#!/usr/bin/env bash

start_time=$(date +%s)  # 开始时间

code_path=$(cd "$(dirname "$0")"; pwd)  # 脚本所在目录
# reduc_path=`dirname ${code_path}`  # 脚本目录的上一级为数据处理总目录
reduc_path=$(pwd)  # 当前目录作为数据处理总目录
# **读取配置文件**
config_file="${reduc_path}/config.yaml"
if [[ ! -f "$config_file" ]]; then  # 检查配置文件是否存在
    echo "Config File [${config_file}] is Not Exist!"
    exit 1
fi
# **解析配置文件（使用 -r 确保输出纯文本）**
target_nm=$(yq eval '.target_nm' "$config_file")
raper_chos=$(yq eval '.raper_chos' "$config_file")
echo "target_nm:       ${target_nm}"
echo "raper_chos:      ${raper_chos}"
# **解析配置文件中的 data_path，支持多个路径**
data_paths=()
for line in $(yq eval '.data_path[]' "$config_file" | sed 's/^ *- *//' | awk NF); do
    data_paths+=("$line")
done
echo "data_paths:"
for dp in "${data_paths[@]}"; do
    echo "  $dp"
done
# **解析配置文件中的 IMASTK 数据路径**
imastk_paths=()
for line in $(yq eval '.imastk_path[]' "$config_file" | sed 's/^ *- *//' | awk NF); do
    imastk_paths+=("$line")
done
echo "imastk_paths:"
for dp in "${imastk_paths[@]}"; do
    echo "  $dp"
done
# **定义输出目录结构并创建需要的目录**
proc_path="${reduc_path}/proc"  # 数据处理过程文件存放目录
echo "reduc_path:      <$reduc_path>"
echo "proc_path:       <$proc_path>"
# res_path="${reduc_path}/res"  # 结果存放目录
mkdir -p "${proc_path}"

update_list="$reduc_path/update.lst"
touch "$update_list"

# **Step 1. 收集所有 SUBSOLAR 的最早 DATE-OBS**
collect_orbit_to_date() {
    declare -gA orbit_to_date  # 使用关联数组记录 SUBSOLAR（轨次） → 最早日期
    echo "-----------------------------------------------------"
    echo "🔍 Collecting SUBSOLAR --> Earliest DATE-OBS mapping..."
    # echo "qwrwesfd ${data_path}"
    local all_exist=1
    for dp in "${data_paths[@]}"; do
        if [[ ! -d "$dp" ]]; then
            echo "❌ data_path 目录不存在: $dp"
            all_exist=0
        fi
    done
    if [[ $all_exist -eq 0 ]]; then
        return 1
    fi

    # 合并所有路径下的 fit 文件
    # 只读取形如 2025-09-08 结构目录下第一级的 fit 文件：
    # 比如我提供了目录/home/vxpp/program/data/L2/imacal_vt/GRB_250205A/2025-02-06，就只读取它下面第一级的目录；
    # 如果我提供的目录为/home/vxpp/program/data/L2/imacal_vt/GRB_250205A/，就读取它里面的日期目录下的fit目录
    fit_files=()
    for dp in "${data_paths[@]}"; do
        if [[ "$dp" =~ /[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
            # 是日期目录，直接读取第一级 fit 文件
            while IFS= read -r f; do
                fit_files+=("$f")
            done < <(find "$dp" -maxdepth 1 -type f -name "*.fit")
        else
            # 不是日期目录，查找其下所有日期目录
            for date_dir in $(find "$dp" -maxdepth 1 -type d -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}$'); do
                while IFS= read -r f; do
                    fit_files+=("$f")
                done < <(find "$date_dir" -maxdepth 1 -type f -name "*.fit")
            done
        fi
    done
    echo "共找到 FIT 文件数量: ${#fit_files[@]}"

    for fit_file in "${fit_files[@]}"; do
        # echo "📄 $fit_file"
        line=$(gethead "$fit_file" "DATE-OBS" SUBSOLAR)
        date_obs=$(echo "$line" | awk '{print $(NF-1)}' | tr -d "'\"")
        subsolar=$(echo "$line" | awk '{print $(NF)}')
        if [[ -z "$date_obs" || -z "$subsolar" ]]; then
            echo "⚠️ Missing header information: $fit_file"
            continue
        fi
        orbit=$(printf "%05d" $subsolar)
        date=$(echo "$date_obs" | cut -d'T' -f1)
        if [[ -z "${orbit_to_date[$orbit]}" || "$date" < "${orbit_to_date[$orbit]}" ]]; then
            orbit_to_date[$orbit]="$date"
        fi
    done
    # **按轨次排序输出**
    echo "🔍 Collected orbit mapping (sorted by orbit):"
    for orbit in $(echo "${!orbit_to_date[@]}" | tr ' ' '\n' | sort); do
        echo "$orbit -> ${orbit_to_date[$orbit]}"
    done
}

# **Step 2. 按轨次建立目录并处理对应的 FIT 文件**
process_each_orbit() {
    for orbit in $(echo "${!orbit_to_date[@]}" | tr ' ' '\n' | sort); do
        echo "-----------------------------------------------------"
        date="${orbit_to_date[$orbit]}"
        orbit_dirnm="${date}-${orbit}"
        # echo "Processing orbit directory: $orbit_dirnm"
        if grep -Fxq "$orbit_dirnm" "$update_list"; then
            echo "✅ Already processed: $orbit_dirnm"
            continue
        fi
        echo "🚀 Processing: $orbit_dirnm"
        proc_orbit_path="${proc_path}/${orbit_dirnm}"
        mkdir -p "$proc_orbit_path"
        > "$proc_orbit_path/allfit.lst"
        for fit_file in "${fit_files[@]}"; do
            if [[ "$fit_file" == *"_${orbit}_"* ]]; then
                echo "$fit_file" >> "$proc_orbit_path/allfit.lst"
            fi
        done
        # 调用 aphot 函数处理该轨次的 FIT 文件
        echo "📊 Running aphot for orbit: $orbit_dirnm"
        aphot "$proc_orbit_path" "$config_file" "$code_path"

        echo "$orbit_dirnm" >> "$update_list"
        echo "✅ Completed : $orbit_dirnm"
    done
}

aphot() {
    local proc_orbit_path="$1"
    local config_file="$2"
    local code_path="$3"

    cd "$proc_orbit_path" || return 1

    # 只检查文件是否存在且非空
    check_filelist() {
        local filenm="$1"
        [ -s "$filenm" ] && return 0 || return 1
    }

    # 封装：执行 Python 脚本 + 检查文件
    run_python_step() {
        local checkfile="$1"  # 需要检查的文件（如fit列表）
        shift
        local script_path="$1"  # 要执行的Python脚本路径
        local script_name
        script_name=$(basename "$script_path")  # 只保留脚本名
        # 检查文件是否存在且非空
        if ! check_filelist "$checkfile"; then
            echo "Error [${script_name}]: ${checkfile} is empty, Exit..."
            cd - > /dev/null
            return 1
        fi
        # 执行Python脚本，传递所有参数
        python "$@"
    }

# **判断是否imastk目录，只做像素坐标测量**
    if [[ "$(basename $proc_orbit_path)" == "imastk" ]]; then
# # ---------------------------------------------------------------------------------------------------------
        # **测量像素坐标**
        cp allfit.lst suc.lst  # 复制文件列表到 suc.lst
        rm -rf err2_finds.lst *_pos.csv
        run_python_step "suc.lst" "${code_path}/find_stars.py" suc.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
    else
# # ---------------------------------------------------------------------------------------------------------
        # **检查关键字**
        rm -rf err1_check.lst
        run_python_step "allfit.lst" "${code_path}/check_fits.py" allfit.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
        # **测量像素坐标**
        rm -rf err2_finds.lst *_pos.csv
        run_python_step "suc.lst" "${code_path}/find_stars.py" suc.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
        # **检查suc.lst长度**
        if [[ ! -s suc.lst || $(wc -l < suc.lst) -lt 3 ]]; then
            echo "[WARN] suc.lst 文件过短，跳过后续步骤。"
            cd - > /dev/null
            return 1
        fi
# # ---------------------------------------------------------------------------------------------------------
    fi
# ---------------------------------------------------------------------------------------------------------
    # **自动找星孔径测光**
    rm -rf err3_aphot.lst *_aphot.parquet *_aphot.csv
    run_python_step "suc.lst" "${code_path}/aphot.py" ${config_file} suc.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
    # **计算极限星等**
    rm -rf *_maglim.csv
    python "${code_path}/cal_maglimit.py" ${config_file} suc.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
    # **计算星等孔径改正值**
    rm -rf *_magc.csv *_magc.pdf *_magc.png
    python ${code_path}/cal_magcor.py "$config_file" suc.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
    # **GRB星表匹配**
    rm -rf grb_xyposs.csv
    # run_python_step "suc.lst" "${code_path}/grb_match.py" suc.lst || return 1
    python "${code_path}/grb_match_mansel.py" ${config_file} suc.lst || return 1
# # ---------------------------------------------------------------------------------------------------------
    # **GRB孔径测光**
    rm -rf grb_aphot.csv
    python "${code_path}/grb_aphot.py" "$config_file" || return 1
# # ---------------------------------------------------------------------------------------------------------
    # **GRB星等孔径改正**
    rm -rf grb_aphot_magc.csv
    python "${code_path}/grb_magcor.py" || return 1
# # ---------------------------------------------------------------------------------------------------------
    # **GRB光变曲线绘制**
    rm -rf *_lc_*.csv *_lc_*.html
    run_python_step "grb_aphot_magc.csv" "${code_path}/grb_lcplot.py" "$config_file" "$r_aper" || return 1
# # ---------------------------------------------------------------------------------------------------------

    cd - > /dev/null
    return 0
}

# **Step 3. 判断有无图像合并的数据，并决定是否处理**
process_imastk_data() {
    # 检查 imastk_paths 是否为空
    if [[ ${#imastk_paths[@]} -eq 0 ]]; then
        echo "❌ No IMASTK paths found in config file."
        return 1
    fi
    imastk_dirnm="imastk"
    proc_imastk_path="${proc_path}/${imastk_dirnm}"
    mkdir -p "$proc_imastk_path"
    allfit_file="$proc_imastk_path/allfit.lst"
    > "$allfit_file"
    found_any=0
    # 合并所有imastk_paths下的*.fit文件
    for raw_imastk_path in "${imastk_paths[@]}"; do
        echo "-----------------------------------------------------"
        echo "🔍 Scanning IMASTK data from: $raw_imastk_path"
        if [[ ! -d "$raw_imastk_path" ]]; then
            echo "❌ Directory does NOT exist: $raw_imastk_path"
            continue
        fi
        cnt=$(find "$raw_imastk_path" -maxdepth 1 -type f -name "SVT*_c_n*.fit" | tee -a "$allfit_file" | wc -l)
        if [[ $cnt -gt 0 ]]; then
            found_any=1
        fi
    done
    if [[ $found_any -eq 0 ]]; then
        echo "❌ No stacked images found in IMASTK paths."
        return 1
    fi
    if grep -Fxq "$imastk_dirnm" "$update_list"; then
        echo "✅ Already processed: <$imastk_dirnm>"
    else
        echo "🚀 Processing all IMASTK data together in: $proc_imastk_path"
        aphot "$proc_imastk_path" "$config_file" "$code_path"
        echo "$imastk_dirnm" >> "$update_list"
        echo "✅ Completed: ${imastk_dirnm}"
    fi
}

merge_subsolar() {
    echo "==== Collect all Light Curves & Plot ===="
    # **收集整理每一轨的光变曲线**
    python "${code_path}/collect_allsub.py" "$config_file" "$update_list" "$proc_path" || return 1
    # **绘制所有光变曲线**
    python "${code_path}/collect_lcplot.py" "$config_file" "$r_aper" || return 1

    # **整理结果并绘图**
    python "${code_path}/get_res.py" "$config_file" || return 1
    python "${code_path}/plot_res.py" "$config_file" || return 1
}

main() {
    # collect_orbit_to_date  # Step 1. 收集所有 SUBSOLAR 的最早 DATE-OBS
    # process_each_orbit     # Step 2. 按轨次建立目录并处理对应的 FIT 文件
    # process_imastk_data    # Step 3. 处理 IMASTK 数据
    merge_subsolar         # Step 4. 合并所有轨次lc csv并统一绘图
}

main  # 执行主函数

end_time=$(date +%s)  # 结束时间
runtime=$((end_time - start_time))  # 计算脚本运行耗时
echo "RUN TIME: $runtime s"
