# R2C3 原生GPC训练长度 × 采样次数学习曲线

同一300更新完整状态恢复，数据/80-20分布/原生网络与损失保持。300的3/16次结果直接引用旧阶段；本阶段只实际测1000/3000/6000。copy-last不重复推理或评分。所有36固定窗、失败和unknown保留；模型时间仅统计。

## 全组灰物体指标

|组|H|完整更新|去噪次数|IoU|灰ROI RGB MAE|轮廓误差px|质心对truth误差px|质心对LAST漂移px|形状结果|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|train_visual|1|—|copy|0.595002|0.095515|0.839404|1.826760|0.000000|{'unknown': 12}|
|train_visual|1|300|3|0.783790|0.048879|0.435081|0.717442|1.475786|{'unknown': 11, 'failure': 1}|
|train_visual|1|300|16|0.772336|0.051544|0.434095|0.732807|1.495634|{'unknown': 12}|
|train_visual|1|1000|3|0.870625|0.030938|0.232103|0.488752|1.579075|{'unknown': 12}|
|train_visual|1|1000|16|0.859016|0.033429|0.247577|0.475905|1.597370|{'unknown': 12}|
|train_visual|1|3000|3|0.932605|0.017233|0.117542|0.187258|1.770831|{'unknown': 11, 'assessable_no_detected_failure': 1}|
|train_visual|1|3000|16|0.934015|0.016691|0.125306|0.211952|1.785270|{'unknown': 11, 'assessable_no_detected_failure': 1}|
|train_visual|1|6000|3|0.938594|0.015772|0.110851|0.207573|1.804461|{'unknown': 12}|
|train_visual|1|6000|16|0.941714|0.015468|0.110189|0.179243|1.804961|{'unknown': 12}|
|train_visual|4|—|copy|0.224879|0.193748|2.968996|6.616078|0.000000|{'unknown': 12}|
|train_visual|4|300|3|0.562509|0.087255|1.085269|2.083026|6.129403|{'unknown': 11, 'failure': 1}|
|train_visual|4|300|16|0.529868|0.089246|1.126151|2.103301|5.985917|{'unknown': 11, 'failure': 1}|
|train_visual|4|1000|3|0.714488|0.057163|0.619374|1.416463|6.283795|{'unknown': 11, 'assessable_no_detected_failure': 1}|
|train_visual|4|1000|16|0.691525|0.062535|0.671202|1.410881|6.201031|{'unknown': 11, 'failure': 1}|
|train_visual|4|3000|3|0.851869|0.029234|0.273046|0.503599|6.483297|{'unknown': 10, 'assessable_no_detected_failure': 2}|
|train_visual|4|3000|16|0.870844|0.026626|0.230743|0.457561|6.613656|{'unknown': 10, 'assessable_no_detected_failure': 2}|
|train_visual|4|6000|3|0.866609|0.027375|0.248652|0.472659|6.518887|{'unknown': 10, 'assessable_no_detected_failure': 2}|
|train_visual|4|6000|16|0.886459|0.024302|0.214720|0.402867|6.494611|{'unknown': 10, 'assessable_no_detected_failure': 2}|
|val_natural|1|—|copy|0.786223|0.050455|0.409067|0.867802|0.000000|{'assessable_no_detected_failure': 6, 'unknown': 6}|
|val_natural|1|300|3|0.819029|0.043760|0.361336|0.447238|0.906056|{'assessable_no_detected_failure': 4, 'unknown': 7, 'failure': 1}|
|val_natural|1|300|16|0.794125|0.047375|0.389742|0.461300|0.840819|{'failure': 5, 'unknown': 6, 'assessable_no_detected_failure': 1}|
|val_natural|1|1000|3|0.843289|0.037477|0.303624|0.346304|0.953002|{'assessable_no_detected_failure': 6, 'unknown': 6}|
|val_natural|1|1000|16|0.833645|0.039376|0.304432|0.349239|0.935964|{'assessable_no_detected_failure': 4, 'unknown': 6, 'failure': 2}|
|val_natural|1|3000|3|0.858406|0.035104|0.272418|0.387656|1.057774|{'assessable_no_detected_failure': 6, 'unknown': 6}|
|val_natural|1|3000|16|0.846808|0.037377|0.286133|0.433309|1.035892|{'assessable_no_detected_failure': 5, 'unknown': 6, 'failure': 1}|
|val_natural|1|6000|3|0.854665|0.037161|0.284919|0.358879|1.033662|{'assessable_no_detected_failure': 6, 'unknown': 6}|
|val_natural|1|6000|16|0.842556|0.039208|0.305674|0.380295|0.999115|{'assessable_no_detected_failure': 5, 'unknown': 6, 'failure': 1}|
|val_natural|4|—|copy|0.570518|0.100618|1.647104|3.467058|0.000000|{'assessable_no_detected_failure': 6, 'unknown': 6}|
|val_natural|4|300|3|0.658888|0.070951|0.740408|1.579941|3.782252|{'failure': 6, 'unknown': 6}|
|val_natural|4|300|16|0.608107|0.076663|0.852982|1.574079|3.451729|{'failure': 6, 'unknown': 6}|
|val_natural|4|1000|3|0.726143|0.060349|0.554988|0.990034|3.739911|{'assessable_no_detected_failure': 4, 'unknown': 6, 'failure': 2}|
|val_natural|4|1000|16|0.704770|0.065378|0.577374|1.003492|3.678346|{'assessable_no_detected_failure': 4, 'unknown': 6, 'failure': 2}|
|val_natural|4|3000|3|0.725701|0.060665|0.567886|1.189436|3.917817|{'assessable_no_detected_failure': 5, 'unknown': 6, 'failure': 1}|
|val_natural|4|3000|16|0.726951|0.061187|0.531723|0.981976|3.738885|{'failure': 3, 'unknown': 6, 'assessable_no_detected_failure': 3}|
|val_natural|4|6000|3|0.727948|0.062156|0.580470|1.190427|3.898065|{'failure': 2, 'unknown': 6, 'assessable_no_detected_failure': 4}|
|val_natural|4|6000|16|0.723843|0.065017|0.578929|1.131382|3.810785|{'assessable_no_detected_failure': 4, 'unknown': 6, 'failure': 2}|
|val_motion|1|—|copy|0.599357|0.094055|0.816104|1.751913|0.000000|{'unknown': 12}|
|val_motion|1|300|3|0.779904|0.051411|0.461324|0.681246|1.594148|{'unknown': 12}|
|val_motion|1|300|16|0.755170|0.054166|0.489163|0.728877|1.487522|{'unknown': 12}|
|val_motion|1|1000|3|0.804387|0.045499|0.389910|0.611734|1.773041|{'unknown': 12}|
|val_motion|1|1000|16|0.797316|0.047158|0.389721|0.556567|1.656401|{'unknown': 12}|
|val_motion|1|3000|3|0.806676|0.044458|0.386624|0.610914|1.841244|{'unknown': 12}|
|val_motion|1|3000|16|0.804188|0.044628|0.389276|0.600437|1.843559|{'unknown': 12}|
|val_motion|1|6000|3|0.820723|0.043690|0.351895|0.555324|1.780229|{'unknown': 12}|
|val_motion|1|6000|16|0.821920|0.043080|0.346986|0.484556|1.746886|{'unknown': 12}|
|val_motion|4|—|copy|0.258134|0.183399|2.518514|5.748023|0.000000|{'assessable_no_detected_failure': 2, 'unknown': 10}|
|val_motion|4|300|3|0.531987|0.098874|1.155297|2.069264|4.748095|{'unknown': 7, 'failure': 5}|
|val_motion|4|300|16|0.489620|0.099890|1.229338|2.208889|4.345090|{'failure': 5, 'unknown': 7}|
|val_motion|4|1000|3|0.595447|0.088481|0.897783|1.793561|5.389747|{'unknown': 7, 'assessable_no_detected_failure': 2, 'failure': 3}|
|val_motion|4|1000|16|0.597446|0.087067|0.855519|1.774623|5.282911|{'failure': 5, 'assessable_no_detected_failure': 1, 'unknown': 6}|
|val_motion|4|3000|3|0.616894|0.083284|0.856886|1.663089|5.765664|{'unknown': 6, 'assessable_no_detected_failure': 2, 'failure': 4}|
|val_motion|4|3000|16|0.610955|0.083704|0.881125|1.793363|5.791970|{'unknown': 7, 'assessable_no_detected_failure': 2, 'failure': 3}|
|val_motion|4|6000|3|0.635105|0.082431|0.795006|1.742399|5.536874|{'unknown': 7, 'assessable_no_detected_failure': 2, 'failure': 3}|
|val_motion|4|6000|16|0.637674|0.081531|0.781935|1.600949|5.637244|{'unknown': 7, 'assessable_no_detected_failure': 4, 'failure': 1}|

## natural非明显8窗与严格近静止4窗

|子组|H|方法|N|灰1−IoU|灰ROI MAE|轮廓px|质心truth误差px|LAST漂移px|形状结果|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
|H1_small_or_nearstatic|1|copy_last|8|0.085219|0.020401|0.208791|0.337151|0.000000|{'assessable_no_detected_failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|1|u300_s3|8|0.138286|0.034654|0.272043|0.373773|0.534669|{'assessable_no_detected_failure': 4, 'unknown': 3, 'failure': 1}|
|H1_small_or_nearstatic|1|u300_s16|8|0.152107|0.037385|0.289289|0.358206|0.509060|{'failure': 5, 'unknown': 2, 'assessable_no_detected_failure': 1}|
|H1_small_or_nearstatic|1|u1000_s3|8|0.114747|0.028868|0.226811|0.338587|0.494818|{'assessable_no_detected_failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|1|u1000_s16|8|0.125605|0.031336|0.236020|0.313052|0.517281|{'assessable_no_detected_failure': 4, 'unknown': 2, 'failure': 2}|
|H1_small_or_nearstatic|1|u3000_s3|8|0.104647|0.027828|0.212295|0.340436|0.585069|{'assessable_no_detected_failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|1|u3000_s16|8|0.117150|0.030854|0.226708|0.357246|0.593544|{'assessable_no_detected_failure': 5, 'unknown': 2, 'failure': 1}|
|H1_small_or_nearstatic|1|u6000_s3|8|0.111167|0.030151|0.234361|0.305880|0.551126|{'assessable_no_detected_failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|1|u6000_s16|8|0.122916|0.032896|0.250409|0.308048|0.565164|{'assessable_no_detected_failure': 5, 'unknown': 2, 'failure': 1}|
|H1_small_or_nearstatic|4|copy_last|8|0.230655|0.056521|0.833712|1.574983|0.000000|{'assessable_no_detected_failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|4|u300_s3|8|0.254734|0.057963|0.512053|0.963973|2.013162|{'failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|4|u300_s16|8|0.304160|0.065778|0.631289|1.137064|1.904001|{'failure': 6, 'unknown': 2}|
|H1_small_or_nearstatic|4|u1000_s3|8|0.186278|0.045741|0.378121|0.829822|1.997864|{'assessable_no_detected_failure': 4, 'unknown': 2, 'failure': 2}|
|H1_small_or_nearstatic|4|u1000_s16|8|0.210451|0.050810|0.425774|0.816955|2.036254|{'assessable_no_detected_failure': 4, 'unknown': 2, 'failure': 2}|
|H1_small_or_nearstatic|4|u3000_s3|8|0.195378|0.047068|0.415608|0.830886|2.016031|{'assessable_no_detected_failure': 5, 'unknown': 2, 'failure': 1}|
|H1_small_or_nearstatic|4|u3000_s16|8|0.207255|0.050333|0.420221|0.735931|2.027578|{'failure': 3, 'unknown': 2, 'assessable_no_detected_failure': 3}|
|H1_small_or_nearstatic|4|u6000_s3|8|0.200841|0.049367|0.438908|0.790515|1.939677|{'failure': 2, 'unknown': 2, 'assessable_no_detected_failure': 4}|
|H1_small_or_nearstatic|4|u6000_s16|8|0.201196|0.051874|0.424760|0.786277|2.012243|{'assessable_no_detected_failure': 4, 'unknown': 2, 'failure': 2}|
|H1_nearstatic_aux|1|copy_last|4|0.006768|0.001160|0.012278|0.059250|0.000000|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|1|u300_s3|4|0.120171|0.031328|0.199604|0.334474|0.286971|{'assessable_no_detected_failure': 3, 'unknown': 1}|
|H1_nearstatic_aux|1|u300_s16|4|0.136665|0.034189|0.217768|0.301386|0.243434|{'failure': 3, 'assessable_no_detected_failure': 1}|
|H1_nearstatic_aux|1|u1000_s3|4|0.083382|0.023879|0.145444|0.308168|0.274833|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|1|u1000_s16|4|0.095641|0.025751|0.147112|0.300413|0.265086|{'assessable_no_detected_failure': 3, 'failure': 1}|
|H1_nearstatic_aux|1|u3000_s3|4|0.084738|0.022579|0.162568|0.264811|0.253974|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|1|u3000_s16|4|0.103098|0.025496|0.184408|0.301347|0.270322|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|1|u6000_s3|4|0.090066|0.024292|0.183932|0.214038|0.240187|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|1|u6000_s16|4|0.099492|0.026186|0.193393|0.259132|0.273623|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|4|copy_last|4|0.007562|0.001390|0.013675|0.063402|0.000000|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|4|u300_s3|4|0.235675|0.056867|0.449523|0.719446|0.665119|{'failure': 4}|
|H1_nearstatic_aux|4|u300_s16|4|0.252964|0.059323|0.472162|0.723622|0.673049|{'failure': 4}|
|H1_nearstatic_aux|4|u1000_s3|4|0.180247|0.044961|0.365597|0.795946|0.747646|{'assessable_no_detected_failure': 3, 'failure': 1}|
|H1_nearstatic_aux|4|u1000_s16|4|0.206610|0.051384|0.406449|0.839767|0.785582|{'assessable_no_detected_failure': 3, 'failure': 1}|
|H1_nearstatic_aux|4|u3000_s3|4|0.159199|0.039545|0.318653|0.502836|0.472790|{'assessable_no_detected_failure': 4}|
|H1_nearstatic_aux|4|u3000_s16|4|0.203014|0.050022|0.403174|0.587634|0.555789|{'failure': 2, 'assessable_no_detected_failure': 2}|
|H1_nearstatic_aux|4|u6000_s3|4|0.154581|0.040378|0.314106|0.446892|0.426894|{'failure': 1, 'assessable_no_detected_failure': 3}|
|H1_nearstatic_aux|4|u6000_s16|4|0.176136|0.047775|0.347553|0.527865|0.504244|{'assessable_no_detected_failure': 3, 'failure': 1}|

## 同权重横向、同采样器纵向逐窗比较

下列为H4全组灰(1−IoU)逐窗赢/平/输；全部H1/H4、每窗delta、natural8/4子组见paired-windows.csv。负相对改善为退步。

|组|比较|赢/平/输|相对灰误差改善|
|---|---|---|---:|
|train_visual|u300_s16_vs_u300_s3|2/0/10|-7.46%|
|train_visual|u1000_s16_vs_u1000_s3|3/0/9|-8.04%|
|train_visual|u3000_s16_vs_u3000_s3|8/0/4|12.81%|
|train_visual|u6000_s16_vs_u6000_s3|9/0/3|14.88%|
|train_visual|u300_s3_vs_copy_last|12/0/0|43.56%|
|train_visual|u300_s16_vs_copy_last|11/0/1|39.35%|
|train_visual|u1000_s3_vs_copy_last|12/0/0|63.17%|
|train_visual|u1000_s16_vs_copy_last|12/0/0|60.20%|
|train_visual|u3000_s3_vs_copy_last|12/0/0|80.89%|
|train_visual|u3000_s16_vs_copy_last|12/0/0|83.34%|
|train_visual|u6000_s3_vs_copy_last|12/0/0|82.79%|
|train_visual|u6000_s16_vs_copy_last|12/0/0|85.35%|
|train_visual|u1000_s3_vs_u300_s3|12/0/0|34.74%|
|train_visual|u3000_s3_vs_u1000_s3|12/0/0|48.12%|
|train_visual|u6000_s3_vs_u3000_s3|8/0/4|9.95%|
|train_visual|u3000_s3_vs_u300_s3|12/0/0|66.14%|
|train_visual|u6000_s3_vs_u300_s3|12/0/0|69.51%|
|train_visual|u1000_s16_vs_u300_s16|12/0/0|34.39%|
|train_visual|u3000_s16_vs_u1000_s16|12/0/0|58.13%|
|train_visual|u6000_s16_vs_u3000_s16|8/0/4|12.09%|
|train_visual|u3000_s16_vs_u300_s16|12/0/0|72.53%|
|train_visual|u6000_s16_vs_u300_s16|12/0/0|75.85%|
|val_natural|u300_s16_vs_u300_s3|1/1/10|-14.89%|
|val_natural|u1000_s16_vs_u1000_s3|2/0/10|-7.80%|
|val_natural|u3000_s16_vs_u3000_s3|6/0/6|0.46%|
|val_natural|u6000_s16_vs_u6000_s3|5/0/7|-1.51%|
|val_natural|u300_s3_vs_copy_last|6/0/6|20.58%|
|val_natural|u300_s16_vs_copy_last|6/0/6|8.75%|
|val_natural|u1000_s3_vs_copy_last|6/0/6|36.24%|
|val_natural|u1000_s16_vs_copy_last|6/0/6|31.26%|
|val_natural|u3000_s3_vs_copy_last|6/0/6|36.13%|
|val_natural|u3000_s16_vs_copy_last|6/0/6|36.42%|
|val_natural|u6000_s3_vs_copy_last|6/0/6|36.66%|
|val_natural|u6000_s16_vs_copy_last|6/0/6|35.70%|
|val_natural|u1000_s3_vs_u300_s3|12/0/0|19.72%|
|val_natural|u3000_s3_vs_u1000_s3|5/0/7|-0.16%|
|val_natural|u6000_s3_vs_u3000_s3|6/1/5|0.82%|
|val_natural|u3000_s3_vs_u300_s3|9/0/3|19.59%|
|val_natural|u6000_s3_vs_u300_s3|8/0/4|20.25%|
|val_natural|u1000_s16_vs_u300_s16|10/0/2|24.67%|
|val_natural|u3000_s16_vs_u1000_s16|6/0/6|7.51%|
|val_natural|u6000_s16_vs_u3000_s16|7/0/5|-1.14%|
|val_natural|u3000_s16_vs_u300_s16|11/0/1|30.33%|
|val_natural|u6000_s16_vs_u300_s16|10/0/2|29.53%|
|val_motion|u300_s16_vs_u300_s3|4/0/8|-9.05%|
|val_motion|u1000_s16_vs_u1000_s3|6/0/6|0.49%|
|val_motion|u3000_s16_vs_u3000_s3|5/0/7|-1.55%|
|val_motion|u6000_s16_vs_u6000_s3|8/0/4|0.70%|
|val_motion|u300_s3_vs_copy_last|12/0/0|36.91%|
|val_motion|u300_s16_vs_copy_last|12/0/0|31.20%|
|val_motion|u1000_s3_vs_copy_last|11/0/1|45.47%|
|val_motion|u1000_s16_vs_copy_last|11/0/1|45.74%|
|val_motion|u3000_s3_vs_copy_last|11/0/1|48.36%|
|val_motion|u3000_s16_vs_copy_last|12/0/0|47.56%|
|val_motion|u6000_s3_vs_copy_last|11/0/1|50.81%|
|val_motion|u6000_s16_vs_copy_last|11/0/1|51.16%|
|val_motion|u1000_s3_vs_u300_s3|10/0/2|13.56%|
|val_motion|u3000_s3_vs_u1000_s3|7/0/5|5.30%|
|val_motion|u6000_s3_vs_u3000_s3|8/0/4|4.75%|
|val_motion|u3000_s3_vs_u300_s3|9/0/3|18.14%|
|val_motion|u6000_s3_vs_u300_s3|11/0/1|22.03%|
|val_motion|u1000_s16_vs_u300_s16|11/0/1|21.13%|
|val_motion|u3000_s16_vs_u1000_s16|8/0/4|3.36%|
|val_motion|u6000_s16_vs_u3000_s16|8/0/4|6.87%|
|val_motion|u3000_s16_vs_u300_s16|12/0/0|23.77%|
|val_motion|u6000_s16_vs_u300_s16|11/0/1|29.01%|

## 固定窗固定噪声H1残差loss

只在1000/3000/6000实测，同一9个batch/36窗及相同noise计划控制条件。核对每batch seed/IDs/shape和generator before/after状态SHA，并使用冻结的同一噪声生成代码；没有保存每个sigma/epsilon张量数值digest，因此不声称独立逐字节核对实际noise tensors。没有测300，曲线不补造300基线；训练batch loss因实际抽样/噪声不同另列。

|累计完整更新|TRAIN展示|natural seen-val|motion seen-val|
|---:|---:|---:|---:|
|1000|0.00131988|0.00164677|0.00184861|
|3000|0.00080521|0.00161387|0.00177633|
|6000|0.00071512|0.00176574|0.00183279|

## 形状、孔洞与碎片

shape unknown保留为未知；可评估未检出失败不是刚性证明。以下固定全36窗，同时列可判定分母和unknown。不同训练点的可判定集合会变化，不能把failure/可判定集合的条件比例变化直接叫保形通过率改善，也不能把unknown从分母删除。所有方法H1/H4孔洞/组件/碎片原始计数和分组汇总见JSON与endpoint-scores.csv。蓝推杆指标独立列在pusher-summary.csv，不能替代灰物体。

|更新|去噪次数|H|固定全窗N|failure|unknown|可判定N|可判定未检出失败（非刚性证明）|
|---:|---:|---:|---:|---:|---:|---:|---:|
|300|3|1|36|2|30|6|4|
|300|3|4|36|12|24|12|0|
|300|16|1|36|5|30|6|1|
|300|16|4|36|12|24|12|0|
|1000|3|1|36|0|30|6|6|
|1000|3|4|36|5|24|12|7|
|1000|16|1|36|2|30|6|4|
|1000|16|4|36|8|23|13|5|
|3000|3|1|36|0|29|7|7|
|3000|3|4|36|5|22|14|9|
|3000|16|1|36|1|29|7|6|
|3000|16|4|36|6|23|13|7|
|6000|3|1|36|0|30|6|6|
|6000|3|4|36|5|23|13|8|
|6000|16|1|36|1|30|6|5|
|6000|16|4|36|3|23|13|10|

unknown的occlusion_ambiguous是一个代理规则：真值或预测灰mask膨胀1图像像素后与各自蓝推杆mask相交，就标为可能接触/遮挡歧义。它不证明物体实际被遮住；可见的轮廓畸变也不能因此算通过。父对1000/原生3次H4逐行核对：24个unknown全部由该代理触发，prediction/truth object-border均False，并非24个实际确认遮挡。评分口径保持冻结不改。

父1000目检发现B双孔恢复、X部分分叉改善，同时O仍圆、C仍明显变形。已有角点/全36窗图库直接引用 父1000视觉核对 (external historical reference: `../parent-view-1000/index.html`)，不重复生成那11张图；本报告48页仅是规定的全部训练点纵向矩阵。

父3000在相同O/B/C/X四窗、固定3次的300/1000/3000对照中观察到B外边更方、X分叉继续改善，O仍缺直角、C仍明显变形。直接引用 父3000视觉核对 (external historical reference: `../parent-view-3000/index.html`) 及其provenance；不重复生成那11张图，也不以四个focus窗替代全36窗统计。

## 证据范围与下一步

横向同update的3vs16变化支持采样器因素；纵向同setting的300→1000→3000→6000变化支持300时训练长度不足的因素，两者可以同时影响。TRAIN改善而val不改善或退步是过拟合迹象，不能用TRAIN单独宣称泛化。nearstatic漂移、形状failure/unknown和局部窗退步须和总体均值一起看。

6000仅是本次观察点，不自动宣称收敛、失败、模型能力上限或继续训练无用。若固定sampler的3000→6000仍持续改善，应建议父审进一步观察10000+；若指标分歧或变化很小，应说明证据不足，不能唯一归因为数据、架构或采样。新阶段最终由父对话安排，实施端不自动续训。

只对当前6个seen字母和固定窗口成立。natural/motion是同12条刻意含运动轨迹且时间窗部分重叠，不能当独立测试或8未见字母泛化。原RGB四步反馈未修色修形、未截幅评分；PNG显示编码限幅时标注。所有36窗H1/H4在48页图库中全量展示。

用户另外指定的O/B/C/X目视主图严格为真实未来/1000/3000/6000四列，固定3次去噪，H1/H4分开；O/C有2行4列大细节。见 四列棱角与孔洞对照 (external historical reference: `focus-gallery/index.html`) 及逐项人工判断focus-gallery/VISUAL_REVIEW.md。16次结果独立保留于完整科学矩阵，不混入指定主图。

真实模型耗时、恢复/检查点/AMP与调用信息见model-work.json及RUNTIME.md；本脚本0模型、0checkpoint load、0冷评分、0physics。通用check_repo历史CRLF/LF问题保持原样，不声称所有仓库检查全绿。


## 6000角点目检与指标分歧

root和报告agent原分辨率独立复核一致：O H1局部恢复但关键直角/方内孔未恢复，O H4仍圆/椭圆环；C H1/H4左臂局部恢复，但底边U弧、下转角和开口两端仍明显变形。B双孔/方边、X分叉改善可见，不等于完全刚性或位姿通过。逐项证据见focus-gallery/VISUAL_REVIEW.md和focus-visual-review.json。

3000→6000固定noise-plan loss：TRAIN 0.000805214→0.000715123下降，natural 0.001613871→0.001765742、motion 0.001776332→0.001832789均回升；同时motion H4 IoU继续改善，natural H4近持平、H1小退。这是泛化差距与指标分歧的线索，不能仅凭TRAIN或H4均值作收敛、已过拟合、继续训练无用或能力上限的结论。若父审进一步观察10000+，应连同验证loss、形状和静止漂移共同判断，不能预设一定获益。
