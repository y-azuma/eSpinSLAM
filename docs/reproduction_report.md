# eSpinSLAM再現実装レポート

- 対象論文: E. Elms, Y. Latif, T.-J. Chin, *Event-based Spinning Object SLAM*, ICCV Workshops 2025
- 再現ブランチ: `codex/paper-faithful-reproduction`
- 評価環境: ユーザー所有のRTX 4070 PC（最終実行時にCPU、メモリ、OS、Python、依存版も記録）
- 詳細な根拠: [`traceability_matrix.md`](traceability_matrix.md)
- 凍結した判断: [`decision_log.md`](decision_log.md)

## この文書の結論

著者実装は公開されておらず、論文だけで一意に再現できない部分が
ある。そのため、本リポジトリは実装全体を「論文に完全忠実」とは
呼ばない。論文または引用文献から忠実に実装できる部分、解釈が必要な
部分、公開データでは評価不能な部分、探索的な結果、論文外の改善案を
以下の5部に分離する。

現在のreference候補は、Claude CodeとCodexが合意した根拠順位と
計算上限に従って実装・レビュー中である。2026-11-24にRTX 4070 PCで
監査済みdev 3 sceneと固定review 18 sceneを実行した。凍結済みD-003は
0/3 sceneで不合格となったため、reference値を調整せず再現失敗として
記録する。Gate Dの相互レビューと全72 sceneの実行は未完了である。

---

## 1. 論文または引用文献どおりに実装できる部分

### 1.1 運動モデルとバックエンド

- 論文Sec. 3.1.2、式(2)-(5)の円軌道、一定回転周波数、一定`R0`を
  `espinslam/geometry.py`に実装する。
- Sec. 4.2.2、式(10)-(11)に従い、保持された各イベントを仮想カメラ
  観測として用い、`f=f*`を固定して`R0`と世界点を最適化する。
- 初回の世界点を原点、`R0=I`とする初期化（Sec. 4.2.3）、新規・更新
  トラックのmulti-view DLT（Sec. 4.2.4）、`N_match`、再投影誤差、軌道
  半径による除外（Sec. 4.2.5）、ランドマーク統合（Sec. 4.2.6）、
  式(12)の忘却（Sec. 4.2.7）を、図3の処理順とともに実装する。
- Table 1に明記された値はreference configへそのまま固定する。
- 論文にない系列末の追加BA、視差閾値、観測の性能目的の間引きは
  reference結果へ入れない。

論文はGTSAMの利用を明記する。直接GTSAM 4.2 `CustomFactor`実装と
in-tree Schur/LMを、結果を見る前に凍結した小規模問題で比較した。

| 項目 | 直接GTSAM | in-tree LM | 差 | 合格上限 |
|---|---:|---:|---:|---:|
| 最終目的関数 | `7.422517109231391` | `7.422517109231421` | 相対`4.07e-15` | `1e-8` |
| `R0` | - | - | `1.71e-11 rad` | `1e-6 rad` |
| 世界点 | - | - | 相対`1.77e-10` | `1e-6` |

全基準を満たしたため、凍結したreference profileでは高速な
`equivalent_custom`を使える。これはこの固定問題に対する同値性証拠で
あり、全入力でのbitwise一致を主張するものではない。直接GTSAM経路も
`backend.solver_backend="gtsam"`として残す。

### 1.2 イベントフロントエンド

eSpinSLAM Sec. 4.1はeSfOフロントエンドをオンライン化したと述べる。
引用元eSfO（arXiv:2405.06216、Secs. 4.1-4.2、Eq. (3)、Table 2）を
一次資料として監査し、次をreferenceへ固定した。

- eFASTコーナー検出（arc/polarityの振る舞はテスト済み。ただし、
  reference [35]の公式実装revision/hash/golden outputを固定できず、
  implementation fidelityはAmbiguous）;
- 密度半径`lambda=7`と、正負極性ごとに独立した平均密度;
- 極性をクラスタ次元に入れないHDBSCAN、`minPts=10`、
  cluster-selection epsilon `5`;
- 最初/最後の`N_sigma=5`イベントの`(t,x,y)`平均をhead/tail記述子と
  すること;
- 前向き時空間半球内の最近傍対応、半径`phi=30`。

オンラインの同時collision処理と時間軸の単位換算は引用元にも書かれて
いないため、第2部の解釈に分ける。

### 1.3 周波数推定

reference modeの式(9)は次の意味で文字どおりに実装する。

1. 全トラックイベントをmidlifeで`A_l`と`A_r`に分割する。
2. `A_l`の全イベントについて、時間シフト後の各点から`A_r`全体への
   最近傍Euclidean distanceを求める。
3. 距離の**和**を目的関数とする。重なる時間範囲だけへの制限、平均化、
   間引きは行わない。
4. 事前固定した全period gridで大域argminを取り、そこから1D ICPで
   精密化する。
5. 直近`aleph`個の有効推定の標準偏差が`σ`未満ならloop closureとする。
   無効な更新はreference履歴を消去しない。

メモリ量を制限するexact batchingは候補、support、reductionを変えない。
総当たり最近傍との距離誤差`1e-12`以内をテストする。旧実装のoverlap
制限、最大イベント数、15%以内なら短い周期を優先する規則、前回値だけ
からの局所探索はreferenceから除外する。

---

## 2. 論文が曖昧で解釈を固定した部分

一次解釈は精度最大化で選んでいない。根拠順位は、式・本文、図・表、
引用文献、一般的慣行、事前宣言したviabilityの順である。代替解釈を
試しても、その結果を理由にprimaryを変更しない。

| 項目 | 凍結した一次解釈 | なぜ曖昧か |
|---|---|---|
| 式(9)の時間座標 | 秒をそのまま使い、時間倍率`1 px/s` | 論文は時間を秒で記すが、pixelとの相対scaleを記さない。 |
| period探索 | `[0.1 s, 観測span]`の幾何gridからargmin、次に1D ICP | 下限、grid、ICP初期化が未記載。 |
| 周波数更新 | `H`更新をtriggerにevent timeで`0.1 s`ごとにbatch | 「Hが更新されるたび」は文字どおりだと計算不能で、batch幅は未記載。 |
| 無効な推定 | 有効推定の履歴へ加えずskip | 「直近aleph回」が無効回を含むか未記載。 |
| robust loss | Huber、`k=2 px` | Table 1はinlier threshold 2を示すが、M-estimator名を示さない。 |
| HDBSCANの時間scale | timestamp secondsに倍率`1` | eSfOもpixelとtimeの相対scaleを示さない。 |
| Eq. (6)の距離空間 | spatial `(x,y)`で半径`lambda=7` | event tupleからの距離としか書かれず、`(x,y)`と`(t,x,y)`を区別できない。 |
| online track collision | deterministic many-to-one | offline eSfOは複数current trackletが1本へ競合する場合を定義しない。 |
| 三角測量の数値破綻 | 前回値、なければ原点へfallbackし、件数/理由をoptimization history・per-sequence CSV・aggregate reportに記録 | 退化時の処理が未記載。 |
| レンズ歪み | Eq. (11)の幾何だけundistort | 論文は入力が事前補正済みかを示さない。 |
| Table 2再投影誤差 | 全inlier observationの平均をprimary、per-landmark平均も併記 | 集約方法が未記載。 |

reference configは
[`examples/configs/topspin_reference.jsonc`](../examples/configs/topspin_reference.jsonc)
である。評価時は`--require-reference`を指定し、凍結profileのどれか1項目
でも異なればpaper-reference出力を拒否する。

`examples/configs/topspin_evk4.jsonc`と
`examples/configs/debris_davis346.jsonc`は過去結果を再生するための
historical experimental profileであり、論文比較には使わない。

---

## 3. 公開データでは再現できない部分

公開TOPSPIN releaseの各sceneにはevents、camera calibration、metadataは
あるが、次がない。

1. authoritativeな真の回転周波数;
2. 著者が評価に使ったCAD model;
3. monocular reconstructionをmillimetreへ固定するauthoritative scale。

したがって、公開releaseだけから計算できない指標は次のとおりである。

- **authoritative-frequency MAE: Unavailable**;
- **true frequency relative error: Unavailable**;
- **CAD ICP RMSE (mm): Unavailable**;
- Table 3のうちfrequency GTまたはCADを要する値。

同じnominal speedの推定値のまとまりは`same-speed consistency`、論文Fig. 6
から目視で読んだ値との差は`figure-proxy comparison`と呼ぶ。どちらも
真値精度ではない。欠測値を0で埋めたり、proxyをGT欄へ入れたりしない。

別途取得したGT/CADによる探索的metricには出所記録を要求する。検証済み
public TOPSPIN runでは、record 10884694がそれらを含まないため、manifestの
`frequency_source`/`cad_source`へauthoritativeと書くだけではtrue metricを
有効にせず、入力監査を失敗させる。将来、著者提供GT/CADまたは未観測の
事前登録データが得られた場合にのみconfirmatory評価が可能になる。

---

## 4. 探索的・回顧的な検証

### 4.1 証拠の位置づけ

- 合成GTは数式、Jacobian、処理順、solver同値性を確認する
  **synthetic verification**であり、論文性能の再現を示さない。
- dev 3 sceneは、論文にも引用元にも値がない必須数値のviability確認
  専用である。過去の探索的TOPSPIN/dev artifactは既に閲覧済みで、以下の
  診断規則は「公開データ未閲覧」を主張するものではなく、最初の監査済み
  reference-profile D-003 dev benchmark前に凍結したものである。frequency
  error、reprojection error、landmark数を最適化しない。生のloop検出は常に
  保存するが、初めて安定条件を満たす
  20推定のすべてが検索端、または1%幾何gridの1刻み以内で
  `span/2`に追従する場合は`loop_artefact_suspected`とし、2/3の
  viability数に含めない。この判定はGTを読まず、推定器の出力も変えない。
- 固定18 sceneは実装レビュー用であり、結果から実装解釈を変えない。
- 公開72 sceneはすでに開発中に閲覧済みなので、最終候補で1回実行して
  も**exploratory / retrospective**であり、held-out、test、confirmatory
  とは呼ばない。

固定18 sceneの一覧は
[`examples/splits/topspin_review18.txt`](../examples/splits/topspin_review18.txt)
に評価前から保存している。dev 3本と既診断の
`soho-perpendicular-fast`を除外する機械的規則で作成した。

### 4.2 `e4c75cd`までの旧TOPSPIN結果

旧実装では、72/72 sceneでloop closure、68/72 sceneでvalid
reconstruction、物体別reprojection error 3.9-4.8 px、same-speed
medianの±5%以内49/72 scene、約2倍のfrequencyを出したsceneが17本、
scene event rate以上のthroughputが9/72 sceneと報告された。

これらは、次の現在reference外の処理を含む。

- Eq. (9)のsupport制限とevent subsampling;
- near-best候補から短いperiodを選ぶharmonic preference;
- paperにないparallax guard;
- 系列末のoffline BA;
- 合成または公開データの性能を見て決めたfrontend値。

したがって旧数値は**方針変更前の探索的・回顧的な参考値**であり、
現在のreference実装の評価結果でも、論文再現の成功証拠でもない。
旧出力を新reference出力と同じ表へ混ぜない。

### 4.3 新しい4070PC評価プロトコル

1. Zenodo recordが示すarchive sizeとMD5を全sceneで検証する。
2. `topspin_reference.jsonc --require-reference --require-verified-topspin`を
   用い、変換済みevents/calibration/metadataのSHA-256も検証する。
3. まず1 dev sceneで実行時間を測り、実行hostのCPU、logical core数、
   RAM、OS、Python、package lock、GPUを記録する。
4. review中は固定18 sceneをscene単位のcheckpoint/resumeで実行する。
5. Gate A-DがClaude Code/Codex双方で承認された最終候補だけ、72 sceneを
   1回実行する。
6. 72 sceneの予測が48時間超、または1 sceneが2時間超なら、近似へ
   置換しない。実測と見積りをユーザーへ報告し、固定18 sceneを
   computational reproduction limitとして正式範囲にする。

実行コマンド:

```bash
python scripts/evaluate_topspin.py \
  --config examples/configs/topspin_reference.jsonc \
  --data-root datasets/topspin \
  --manifest datasets/topspin/manifest.csv \
  --output-dir runs/topspin_reference_review18 \
  --review-subset --require-reference --require-verified-topspin
```

scene checkpointはmanifest row、setting、完全なconfig、入力file hash、
Git commit/source-tree hash、dependency lockが一致し、必須artifactのhashも
一致するときだけ再利用する。output directoryのrun fingerprintが異なる
場合は既存結果を削除せず明示的に拒否する。完了・中断scene、raw
diagnostic、hardware provenanceを保存する。

固定18は実装レビュー専用で、論文Table 2との比較表を生成しない。最終の
clean commitで監査済み72 sceneを厳密に選んだ場合だけ
`--paper-comparison`を指定できる。論文表との集計は全72 sceneを使い、
viabilityに使用済みの3 dev sceneを除く69 sceneの表もリーク監査用として
別途残す。どちらも公開済みデータの回顧的・探索的結果である。
比較対象にできる実行経路は現時点で`eSpinSLAM`本法と、周波数を固定した
`No landmark association`だけである。FFTは論文から一意に実装できず、
free-frequency custom solverはdirect GTSAMとの同値性が未検証なので、
設定値がreference configと一致しても論文比較から除外する。公開TOPSPIN
の取得監査はfrequency真値や著者CADを含まないため、manifest文字列だけで
それらをauthoritativeへ昇格させない。

### 4.4 新reference評価結果

2026-11-24、まずcommit `34f18493ba1d0f37da317445bd405ca7635b8c5a`を
clean treeで監査実行した。その後、手法・設定・入力を変えず、loop確定時の
周波数と初回BA内訳を保存するreporting-only変更
`6b4868a474fc2b79eed66552d11b14811a0aecd8`で、同じRTX 4070ホスト上の
dev 3 sceneとfixed review 18 sceneを再実行した。以下の表、内訳、実行時間は
この修正版再実行の値であり、最初の`34f1849`実行の結果と混在させない。
入力はZenodo record 10884694から取得・変換し、
archive size/MD5、変換後file SHA-256、calibration、resolution、object、
splitを`--require-verified-topspin`で再検証した。method configは
`--require-reference`で完全一致を要求した。

実行環境はUbuntu 24.04系（kernel 6.8.0-149）、x86_64、16 logical CPU、
RAM 33,475,567,616 bytes、RTX 4070 12,282 MiB、Python 3.10.14である。
最初の`34f1849`実行のsource tree SHA-256は
`16ab88a6b085fb9aabdf3cb9cddb7273e1c6c7f297e4963789e4ad29a487894c`、
修正版`6b4868a`再実行のsource tree SHA-256は
`206875ef7da611d9e614b7e73126375b42b1ab28c8b1c1e46769389150e7fcc2`、
dependency lock SHA-256は
`3e90af4a5ed0c2cc58f3250e2c68394fa785530d4016b017266e1b2927859d1e`である。
両source treeの差はreportingとそのテストで、reference手法コード・設定値・
入力sceneは同一である。

#### D-003 canonical dev 3 scene

- reporting監査修正版での再実行時間: 120.09 s;
- loop検出: 3/3;
- 事前登録したloop artefact疑い: 0/3;
- 最終inlier landmark: 全sceneで0;
- 最終BA status: `not_run` 1件、`converged` 2件;
- D-003 eligible: 0/3（必要数2）、`passed=false`。

失敗はloop確定前の周波数帯ではなく、その直後の最初のBAから始まった。

| Scene | loop確定時の`f*` (Hz) | 初回BA後の残存track | reprojection除去 | radius除去 | 最終online推定 (Hz) |
|---|---:|---:|---:|---:|---:|
| Hubble perpendicular slow | 0.073685 | 2 / 91 | 81 | 8 | 7.229254 |
| TDRS side-on medium | 0.139795 | 3 / 127 | 106 | 18 | 10.0（探索端） |
| Camera top-down fast | 0.129476 | 1 / 184 | 1 | 182 | 4.894419 |

loop確定時の3値は論文Fig. 6から読める約0.04--0.16 Hzの帯域内である。
最初のBAは`converged`したが97--99%のtrackを除去した。その後は、論文に
記載された処理順どおり、痩せた`H`でEq. (9)を再推定し、有効推定の大半が
消失、残る推定は高周波側へ移動した。高い`f`を使うEq. (12)のforgetが
さらに`H`を削り、回復しなかった。`frequency_est_hz`はこの「最後に有効
だったonline推定」であり、loop確定値は`frequency_loop_closure_hz`へ別に
保存する。

最初のBAが適合しなかった根本原因は未特定である。候補は、固定した`f*`の
分解能、秒単位へ適応したfront-end trackの品質（`phi=30`、many-to-one
associationを含む）、Camera sceneにおけるSec. 4.3の縮退であり、いずれも
論文から一意に再現できないAmbiguous項目である。Hubbleの確定窓は
`T*=13.00 -> 13.57 s`、`T*/span=0.58 -> 0.54`と緩やかにdriftしており、
凍結済みの`span/2 +/- 1%`診断では疑い扱いにならない。この規則を結果確認後
に変更しない。

したがって、凍結一次解釈は公開TOPSPIN上でD-003の最低viabilityを
満たさなかった。これは論文手法そのものの反証ではなく、未公開著者実装を
曖昧な記述から再現した今回の一次解釈の**reproduction failure**である。
D-011に従い、代替frontend定数の探索、harmonic guard、parallax guard、
terminal BA等をreferenceへ追加しない。原因切り分け実験はreferenceと分離した
Future Workとする。

#### Fixed review 18 scene

- reporting監査修正版での実行時間: 691.61 s（11分32秒）;
- loop検出: 12/18（66.67%）;
- valid reconstruction: 0/18;
- reprojection error: 全objectでUnavailable;
- CAD ICP RMSE: author CAD/scaleが公開されないためUnavailable;
- loop確定時のfrequency: loop検出12件で0.0360--0.1447 Hz;
- 最後に有効だったonline estimate: loop検出12件で0.0372--10.0 Hz;
- loop確定値によるsame-speed consistency median absolute deviation: 2.92%;
- mean throughput: 734,302 events/s。

same-speed consistencyは同じnominal speed内のまとまりであって、真のfrequency
MAEではない。旧20.45%は初回BA後に`H`が崩壊したあとの最終online値から
計算されており、この目的には無効だった。修正版はloopを確定した時点の
`frequency_loop_closure_hz`だけを使う。公開releaseにauthoritative frequency
labelがないため、修正後の2.92%からも周波数精度の成否を主張しない。18
sceneで再構成が1件も成立しなかったことは、D-003失敗と整合する
reference-pathの観測結果である。

また、2.92%という群内medianは一部の大きな不一致を隠す。loop検出12件の
うち、`camera-perpendicular-med`、`tdrs-perpendicular-slow`、
`tdrs-topdown-med`のloop確定値は各speed群中央値のおよそ0.5倍、1.86倍、
1.98倍だった。このため、2.92%だけを全sceneの良好な一致と解釈しない。
同じ入力・コードでも、別ホストとの浮動小数点差によりBA反復数や初回BAの
除去理由内訳が境界で僅かに変わることを確認した。結論は変わらないが、
以後の全72 sceneは同じRTX 4070ホストでのみ実行する。

review 18の実測から72 sceneは約46.1分と予測され、D-008の48時間上限内で
ある。ただしD-008に従い、全72 sceneはGate A--DをClaude CodeとCodexが
相互承認した最終候補でのみ1回実行するため、現時点では未実行である。
機械可読な集計、per-scene row、run identityは
[`evidence/topspin_reference_20261124/`](evidence/topspin_reference_20261124/)
に保存する。旧`e4c75cd`の値は混在させない。

---

## 5. Future Workと実験的解釈

次は合理的な改良候補だが、論文に記載されないためreference再現へ
入れない。

- 対称物体で基本周期とsubharmonicを区別するharmonic guard、候補確率、
  ambiguity/abstention;
- Eq. (9)をoverlap範囲へ制限する、距離を平均する、eventを間引く等の
  approximate variant;
- near-best候補から最短periodを選ぶ旧15% rule;
- triangulation parallax guardとdefensive/safe backend;
- terminal offline BA;
- greedy one-to-oneまたはHungarianによるonline track association;
- 全段でのundistortion;
- 論文Table 3のFFT詳細が引用元からも一意にならない場合の複数解釈。

これらを実装・評価する場合は、別config、別output directory、別表を
使い、reference結果より高い性能が出てもreference既定を変更しない。
改善版の調整に公開TOPSPINを使った場合、その結果も回顧的探索として
表示する。

### レビューとmainへの統合条件

mainへ統合する前に、以下をすべて満たす。

1. Gate A-DをClaude CodeとCodexが個別にレビューし、blocking指摘を
   解消する。
2. lock環境のportable testとoptional GTSAM gateを記録する。
3. traceability matrix、decision log、本レポート、READMEを一致させる。
4. referenceとexperimentalの出力を機械的に分離する。
5. 実装コード、評価条件、4070PCの結果をClaude Codeへ再提示し、
   コードと結果の整合性を双方で承認する。
6. ユーザーの最終承認を得る。

性能が悪いこと、公開TOPSPINでharmonic errorが出ること、計算が遅い
ことだけを理由に、論文にない処理をreferenceへ追加しない。
