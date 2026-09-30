# Phase 24D — Primary live K=16 collection

**Status:** `phase24d_primary_live_dataset_complete_awaiting_discovery_analysis_authorization`

PHASE 24D COLLECTED THE FROZEN K=16 LIVE-ACTIVATION DATASET AND RESPONSE-LEVEL LABELS ONLY. NO PHYSIOLOGY PREDICTOR, AUROC ANALYSIS, LAYER×TOKEN SEARCH, SAE ANALYSIS, CAUSAL INTERVENTION, OR LOCKED-TEST SCIENTIFIC ANALYSIS WAS PERFORMED. LOCKED TEST OUTCOMES REMAIN SEALED.

## Phase-24C verification

Verified: **True** · selected_k check: **True**

## Model / generation

- Mistral: `mistralai/Mistral-7B-Instruct-v0.2` @ `63a8b081895390a26e140280378bc85ec8bce07a`
- BF16 · sdpa · T=1.0
- max_new_tokens=200 · batch=1
- Capture loop: code-equivalent to Phase 24B (`_generate_one` SHA match; canary not re-run)

## Schedule

- Reused Phase-24B: **96**
- New requested: **528**
- New completed: **528**
- Final total: **624**
- New schedule SHA256: `e8f8302fe693d49662fbba27a3057af55a5d1ae0881a1cef47b71895caf637d5`
- Generated tokens (new): **40046**

## Integrity gates (528 new)

```json
{
  "gates": [
    {
      "gate": "G1_completion_528",
      "observed_completed": 528,
      "observed_requested": 528,
      "pass": true,
      "threshold": 528
    },
    {
      "gate": "G2_alignment_100pct",
      "observed": 528,
      "pass": true,
      "threshold": 528
    },
    {
      "gate": "G3_activation_zero_nan",
      "observed_nan_count": 0,
      "pass": true
    },
    {
      "gate": "G4_logit_zero_nan",
      "observed_nan_count": 0,
      "pass": true
    },
    {
      "gate": "G5_artifact_integrity",
      "observed": 528,
      "pass": true
    },
    {
      "gate": "G6_no_replacement_seeds",
      "observed": 528,
      "pass": true
    }
  ],
  "passed": true
}
```

## Activations / logits

- Activations stored: ['embedding_pre_block_residual (bf16 bits as uint16)', 'post_block_residual layers 0..31 (bf16 bits as uint16)', 'final_rmsnorm (bf16 bits as uint16)']
- Logits stored: float16 full vocab vector per prediction step
- New storage bytes: 10635849256
- Mean bytes/traj: 20143653.893939395

## Gemma grader

- `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`
- Graded newly generated responses only; Phase-24B labels retained via provenance

## TRAIN realized yield

Totals H/A/D: **155** /
**45** /
**120**

≥2H∧≥2D prompts: **12** / 20
(gate ≥10: **True**)

| prompt | H | A | D | ≥2H≥2D |
|---|---|---|---|---|
| `roleplay_091` | 2 | 0 | 14 | yes |
| `roleplay_224` | 5 | 0 | 11 | yes |
| `roleplay_078` | 0 | 0 | 16 | no |
| `roleplay_230` | 5 | 8 | 3 | yes |
| `roleplay_192` | 10 | 6 | 0 | no |
| `roleplay_163` | 11 | 0 | 5 | yes |
| `roleplay_030` | 14 | 2 | 0 | no |
| `roleplay_370` | 15 | 1 | 0 | no |
| `roleplay_036` | 2 | 1 | 13 | yes |
| `roleplay_236` | 1 | 8 | 7 | no |
| `roleplay_116` | 7 | 1 | 8 | yes |
| `roleplay_014` | 12 | 0 | 4 | yes |
| `roleplay_040` | 13 | 0 | 3 | yes |
| `roleplay_003` | 10 | 4 | 2 | yes |
| `roleplay_119` | 16 | 0 | 0 | no |
| `roleplay_207` | 1 | 1 | 14 | no |
| `roleplay_355` | 7 | 7 | 2 | yes |
| `roleplay_296` | 2 | 0 | 14 | yes |
| `roleplay_143` | 12 | 3 | 1 | no |
| `roleplay_308` | 10 | 3 | 3 | yes |

## VALIDATION realized yield

Totals H/A/D: **41** /
**15** /
**72**

≥2H∧≥2D prompts: **6** / 8
(gate ≥4: **True**)

| prompt | H | A | D | ≥2H≥2D |
|---|---|---|---|---|
| `roleplay_026` | 9 | 1 | 6 | yes |
| `roleplay_122` | 0 | 0 | 16 | no |
| `roleplay_304` | 4 | 0 | 12 | yes |
| `roleplay_263` | 8 | 0 | 8 | yes |
| `roleplay_281` | 7 | 5 | 4 | yes |
| `roleplay_366` | 11 | 1 | 4 | yes |
| `roleplay_123` | 2 | 0 | 14 | yes |
| `roleplay_009` | 0 | 8 | 8 | no |

## LOCKED TEST (sealed)

```json
{
  "artifact_hashes": {
    "labels_SEALED.json": "91ce78a009927cd6cf0b8cc5449bafab4bbf6018b4598507a747ad4633828df3",
    "new_trajectories_SEALED.jsonl": "88a1ed7a7dd030c39d0e80d1d189a4fdfd7846b52847113befe22f2887689b73",
    "trajectory_artifact_sha256s": {
      "roleplay_007__r00": "b5befd3b302c4be466f5ed55cc4f16dd1c4883d755e16593448808aa44955836",
      "roleplay_007__r01": "33ccdae39ae5008975c498e4e730c687ae5f2697ad8b21bf9c2a9b0bca93857d",
      "roleplay_007__r02": "8c721d5e06589478c0104d827cb1d5693a47995d5e25375fe1473589f1d788c8",
      "roleplay_007__r03": "3c05e97031eb2996da7da9f69aa6a05befc649dd63c2896715f1f796e39e437f",
      "roleplay_007__r04": "2c1d1d5b3257a8580fb177d72e1d34aea90c24121c12249dbdeb7dd0fdf937d3",
      "roleplay_007__r05": "1323311f550d66d6d5adc3761f71b5c7a6a632ef41c8e66593ac3e630e7f9e05",
      "roleplay_007__r06": "68c8e733713b480ae3789dbe5ffafd9674516317414a5ae0a8b0456d80e37157",
      "roleplay_007__r07": "146930df9a53923ae8817f67f18f78431e9499d6780aa97624a73c4c6d0559fe",
      "roleplay_007__r08": "e4868a0a004f6bfa44fb5bf50aaad586bd1c8a9f2fb04665ca7dd2771075d6d5",
      "roleplay_007__r09": "d707aafd5688fda3ce8b5c476980b4ba3f4b56dfb2edba446a2f3999053cd24a",
      "roleplay_007__r10": "f3a84115530ba00e552b7c89defb8a2ef043e98cca5861ee29ed414a3ce022ed",
      "roleplay_007__r11": "0a6cecf890115b589213e68889fb43210e906bf0998909abc70e448fd99e47e2",
      "roleplay_007__r12": "2ad7cd5326a57a4eeabc59f73ce121514c00a2af0ab3fb7651ff6d2d2378340d",
      "roleplay_007__r13": "ec592eefd99ecebefe56065d3c74b7bb60c100c987494c11a2d9bf52f53747a0",
      "roleplay_007__r14": "7aeb4dec4f99d4f5ecb9713cd89715cb31a05a7ddd3a18fb2cb89d958bbf22a2",
      "roleplay_007__r15": "d1ecd86df70200e4d42fccf7ab9ddd72363265db3fa40565becf07c492c1d51e",
      "roleplay_111__r00": "dc230edb18e68b897573191ff09f202e21c193c9ae25862bfa3a3ccec9a626b6",
      "roleplay_111__r01": "6c399f581a8335552d03f3a398590c904aacb073875ec4c5ae9cd3f1dbd1f342",
      "roleplay_111__r02": "c0e597a9af0c243c70d7df0378fccc3827b92ebc45dfaf83b562369a8c7f3b28",
      "roleplay_111__r03": "cb6da6bed731cb53584fb8967bd1a28ff497f6cf3e83b0c20314c08d2a9b227e",
      "roleplay_111__r04": "6957611c786b869f13f8744fc5a7f0404b725deead78c130769fcfb34e4ec276",
      "roleplay_111__r05": "3f57356a3371f284313b185b5db44159852df210a8c178f638e8ba8d8c84d649",
      "roleplay_111__r06": "6ede3caf1c28df7e63df318fd8108b91498ce5d58422330885e206caafbb53d2",
      "roleplay_111__r07": "50752fb389b85c996edfef576a2e15ea7e02661cbad1ad874048e7fe787b5955",
      "roleplay_111__r08": "ab84cf588bdd85e1e07e0f5b1c2c137ce0378c9e9f89f387aeb6d0f3d23627e2",
      "roleplay_111__r09": "ab90d21fa89f0c9e12e050a9f59e13fdb798acc856ed013f5d079cb1d5b03c05",
      "roleplay_111__r10": "618e46b58bcfa5cfdc03fefb909daf17ae4acacef4fbadaeffa5ac13d5c1f3a3",
      "roleplay_111__r11": "f7e0ed5ca87618ab9d4eab3dcbb3f7c6e11c84a20128897a8aaef55a2fd0a83f",
      "roleplay_111__r12": "6ede3caf1c28df7e63df318fd8108b91498ce5d58422330885e206caafbb53d2",
      "roleplay_111__r13": "3b60def131c448d8c9f5a844290005a8a10e7e280a618b1afb4ee114031a09a0",
      "roleplay_111__r14": "3819e5c186c80281128ab4ff28ceca771cd30b063833554c3733f64c90dd6735",
      "roleplay_111__r15": "8858740b7abbc972a4c67dda4c1245fb16daf894e63271d94516a9077e1bfae0",
      "roleplay_174__r00": "0f93ebcd4496c29346fd594e349c85b3ab38f5dd669b82516060e831a55ae769",
      "roleplay_174__r01": "219b37ba54716447029b157ba724f42d854be3aa905db1eb124f6de82a95e67d",
      "roleplay_174__r02": "9fde5aa30de910c40e2a2ebb7c81590c56ea36ff8de0180ad4038da07b6b057b",
      "roleplay_174__r03": "4b0faca3ae5497e4b6ed6a30ccfaaa37bca9459d9f3b2ff0b512cad820c1e9af",
      "roleplay_174__r04": "8ffab3cb2d2fd70be35c9433e07145506dbbdf7a17c8ee176382dc404b8587ba",
      "roleplay_174__r05": "83ab4e97ac1b517878d565353349d03b7d6b340ed643e0689524d1cd2314af74",
      "roleplay_174__r06": "df1f8c7b0f0967f33f2d820f92b3bc7c150e71b98949f726f2b135a15087272d",
      "roleplay_174__r07": "44d25782586c6c6ed497b1510786e4f7a998593a55081db650a82cf4916ac4a1",
      "roleplay_174__r08": "966101418786fbd3133ac68228af66ce6d843308698ebf853b6b3bc6f4d7c558",
      "roleplay_174__r09": "e1e0fb6c8e34eb61f302f64186293512fdc09eca66872eee931b78a281f43725",
      "roleplay_174__r10": "e79bdc75ca2768d54b9426fe7ac8210aa342ebb5fc0fe2478286086d56ab8b78",
      "roleplay_174__r11": "14cb6edd62cf6774134a81c72fae7404606373e2851ed625f229173dda183bc8",
      "roleplay_174__r12": "4508249c79068451fd5120ba8c69f9ddb2e096c896c327142c1aff781189c669",
      "roleplay_174__r13": "2ae2779e511cb1e572ea9708c8cd5fec1c716a14d4af56d60827c956e282c560",
      "roleplay_174__r14": "30ccf565eea77559871d81fec9d416d17f05a4d7aeef9358df79ab46c0b8b880",
      "roleplay_174__r15": "2a45e53bc2f77b9272c74a6abc13b2763f1b830e37560bfe7e1195886ce0dc32",
      "roleplay_193__r00": "aad5f41fc38524f4e811a2d23c899a478fd1cb127df825a664182fd546f37119",
      "roleplay_193__r01": "0f9e4d8a4482ec64409d35f1df707321b7459a1dcceb40f2aee54864f69e716d",
      "roleplay_193__r02": "b3ecaba712915d79b54dcea061ce38c2d4caad7fba840392f9830d0dfca38466",
      "roleplay_193__r03": "a62ae4261d9f5f14eba3f27838599d792f2263c4128ef12b0e61650d4064d4e1",
      "roleplay_193__r04": "86033461f8f9218b3a03b40817e873c5e4184b6512dc90180eeda3e4850e25ca",
      "roleplay_193__r05": "a6398dbd0bc421f7aa0275e507ef1469b2e8c5afeedcea7c960caf029ed6d747",
      "roleplay_193__r06": "ab0906bcc1d184409fa37c033b56d2ab853f4241b4d1527193abfe054ddbe53c",
      "roleplay_193__r07": "13c0dd375923ac3ff11aeb3d329dfd375dd2205262c37bb0762f1bad6d77325a",
      "roleplay_193__r08": "d69f2600e197cf3ddfe8f3473af495db89c2490052cca876078d5924c8a9ffbd",
      "roleplay_193__r09": "f8ab31ce41488a9533da5237ef812746952db292195691d99206ccbb7b4f31d6",
      "roleplay_193__r10": "a46ff47a052d6f696be57bab41bc2b519ffc327e8317e85b5930e0055c15573b",
      "roleplay_193__r11": "5e4e5cc31a77d6077a5b5b3f04a3d7e077c9fc362a8bf2d2d255c19f915fbe48",
      "roleplay_193__r12": "1bbbde6ba137f616764fe7057f3b2a591e3e22825c254e047f22ffc805f2fcdc",
      "roleplay_193__r13": "ad0fb1639cafe46ba13af28244522ef0191a0e45c146642a47d0a0256d721ec1",
      "roleplay_193__r14": "afd91cc886152e1625de252b45efa4000f45aef7bc8d6c21ff47b496ae4518a8",
      "roleplay_193__r15": "626f3d4e204e8b824b53cecfef87c6b1cdd4c6cf0763bc49c47964d300efc237",
      "roleplay_203__r00": "3bf7d3eee79bb7d1896698cdd78c7e117a977a434c7bc9bf1717fc84c5ea25ee",
      "roleplay_203__r01": "92e152f2ea6841cab283d497c4172b1ea784395a596a1d68813e5c37197a005d",
      "roleplay_203__r02": "6f01c94059cd1ef3f8aad7bb07b4ecd90dae3779f2b6803321e12718076722d2",
      "roleplay_203__r03": "3221931e1e8ddd3780c3d178e2d58d139b5b1309d851a49f1380df28c4d75edc",
      "roleplay_203__r04": "4211e059a0f268d3b9623ca8d31d9aaf94a1a64fc32a101db2860648cf3efc51",
      "roleplay_203__r05": "b7f850c0749f354389b341dbe52b4d1e4913e8d0fd6842c2dd70de9fb053ee34",
      "roleplay_203__r06": "213eddaedc83998f3abcd81921985b8114600cc792347515b43f7e4afe0ac4d9",
      "roleplay_203__r07": "3a57a989cd6b3190c872f571686836a2aee48cd221435311aae0b722e1efec1b",
      "roleplay_203__r08": "d994c569527bba4a17c6562cd38de0255027d8a6409aeac51099a804a195ad3f",
      "roleplay_203__r09": "7dd1a2738a8ac7bb30e36e638254060133caf4f76af16a175c866990203130d1",
      "roleplay_203__r10": "cd865e913ff543dda496b7a021540877d77c345a8c3cf366df14ed30dac8560f",
      "roleplay_203__r11": "5f06bb168a78c288bed50f3bc1df47558cc900fc2d859866948baf2ebf90c127",
      "roleplay_203__r12": "99395e820ac1ad7327d9dc3d5b4387ff211e39c31143fe8ba1b0978eb02a0f6f",
      "roleplay_203__r13": "df996f4c881dcf3c4205ca690ee4b3ee0189eed9f1d22536d9c22c12e324b9a2",
      "roleplay_203__r14": "7aa5f6366c7e2b89d0b45823eee89d57b4cca71f119fead74504a68e04f9c687",
      "roleplay_203__r15": "93870e4293ac7dc508f69f95d3df0bbaa1300afd4816b9ef7a2855efe02e69ea",
      "roleplay_206__r00": "743e55945cb4426fa3a72192890be65d5ac00a6f342de0723f18ba0f5bd5ad69",
      "roleplay_206__r01": "711ffe51032ca5073f3b632aee511b18f47fed56975d5abb7f0b2367ad229755",
      "roleplay_206__r02": "1e4f7db29c99e3249a0c15d1dbdbb7e62a7f98417b814c716e6f8a3fdaecf072",
      "roleplay_206__r03": "f438fc31ea6db37cce3947edc55dafa0e8374ef45f1d53f1b54186ed77b6a364",
      "roleplay_206__r04": "be0e292d750586e6645b8d872e76c9b87767e4933f6083b2b9fa3a81e8a492ce",
      "roleplay_206__r05": "0d09b07f28fbf6568411d4d63aeb36158f590fe794e00b1cec9b54dab9414f8a",
      "roleplay_206__r06": "95f6b306b7aeca3da3ebd4ea5de8c2572d231c9f1adb822bc666c1640a1deb50",
      "roleplay_206__r07": "1687b6d2e2b99108c84cff23526c675c22916fd301d931baf3d0afbc8904b426",
      "roleplay_206__r08": "6bce1228ab03daba36d2c111cac9fc2fef445a274e53da003cf63eb16a9b2875",
      "roleplay_206__r09": "ae9881c16bfa02b5ef7df913a5cf16689d9d9a3cc1848a6c0a79b9190cab7ece",
      "roleplay_206__r10": "f6a03580ce042be740fe85102657b76f8e98efda3354e25c554475c744f65fc2",
      "roleplay_206__r11": "5b330b7e0de98b0a8095b43e29e05e020d0e700755a42bee037045d8bb574415",
      "roleplay_206__r12": "0804c8bc39d30b8555c4ef9828882f5f539f52b18b1453fd587fa5c8b9ea2144",
      "roleplay_206__r13": "79e9f0d9fa08c20356de14fa1ba462171b424936f6bf9bf670707a7ba526aa0c",
      "roleplay_206__r14": "9b657cc0920032f4191ea0d4b65a1b12cb0c4d54e4b2ebda03d9245c7552471e",
      "roleplay_206__r15": "b0f8e7588a16ac248061643a9ee68d58cd5ed69bc60c60af03ac3870bb69a0a2",
      "roleplay_245__r00": "1acbf543c024f97a593bdd138c05c49b5ae3fb75abb12b2fda428c97bd28dbd9",
      "roleplay_245__r01": "d04cf71305bc0bd96308ab4f979133a726e116e2e45ef5013dbfb9fd4531eacd",
      "roleplay_245__r02": "258e2565e03fa962cc0ed63b24c8700ccec5a6bdfb9e2bc0530a4a653f88e5e5",
      "roleplay_245__r03": "43446ce632d266887445505e80539c08db09c69078260a26180856a288d5ea08",
      "roleplay_245__r04": "a84e403cbe19ce80b94736b1e39df8595649d9df06a96ae5c9da17873869c91a",
      "roleplay_245__r05": "4ba77c14f514a986925f6fd61e5eb0348b2aa6f1c251f7f533db62070ba6c2ec",
      "roleplay_245__r06": "acdd19a50aa2cfae40f6f377164c671bc95f228b912b55a635ff25d490905125",
      "roleplay_245__r07": "5840b38cb0c2696c6d0720424dfebd6ce901c05576c8dbf6c346749fa6674ec2",
      "roleplay_245__r08": "c5ae8ca716e9b77a01a05a459b415ad89a42092e64ef1bf0d78df689b268a06d",
      "roleplay_245__r09": "b56f257d1f304ed1da252141cf6f845466fa18f58f5769f90a6d41702d14b6f8",
      "roleplay_245__r10": "8fd3932d4185533fbadb73925e2f58a57eada41219ceef925dd4afe1e2b2ae35",
      "roleplay_245__r11": "a804c7472ee5c41a8e18096d230de7460cf7885062169d2146ca5646a880ffca",
      "roleplay_245__r12": "b324e74a37fc89fd2899a72e020b665094dbb067ca72f90b7ef7b63597de40d5",
      "roleplay_245__r13": "7b7e3af28c3e914aa53d9705d5012a079ff8e99fe8a9b78de0a3b67d38656fcd",
      "roleplay_245__r14": "bd35d5b230620e0f9c2be3df0ca370ed232f8d11b18cc29309a93e6012c74a40",
      "roleplay_245__r15": "0fc1b98ee3d472c322d0dd3f651ce90264d7738350e8ffd07ed87924a49a2ee6",
      "roleplay_275__r00": "72a98f7a7ae1e183574e92f5cc5a74ca2bd9d5618206015238b09988ac43eff4",
      "roleplay_275__r01": "e95a385a65bb5c858e713197c23508a972842be87ef20832bc2a92e6a939a04c",
      "roleplay_275__r02": "203e15b0e3851f2a62874d2c8ee4cb103d7953dc6d31f86251acef3a4ccd482f",
      "roleplay_275__r03": "699996aa0a8da41bbcb5a889bc6592319cdb6273795335d65bd17caea6d4fb3b",
      "roleplay_275__r04": "4729b5082de86cada0cb73f78e6205aa4e0adc06560eacbae64585fd77d47474",
      "roleplay_275__r05": "6ad71fa0f6c7ec12c76a5218df2aece5a2a9c5b4f971d9edb3edce0daaed38cd",
      "roleplay_275__r06": "4d736cbdc74700510533487b8308ad06b5b031181c03291e794ee6c3932a7317",
      "roleplay_275__r07": "1167258a9d7b221a9b0f4325d78740d47729c2b1dcac54705264f1764e632497",
      "roleplay_275__r08": "8abe430d97ad8579d444c4e5297b7af9ee5e97b8fcb69eafd43d2a56a7dc7176",
      "roleplay_275__r09": "9314cd323ff22de34fb661d26df6459d8cca6a5824e06f0371d97b346fd94297",
      "roleplay_275__r10": "99896a0645044c74bbc2b31735bd7733ecfc3ebfe713f5a2572dd799ba125836",
      "roleplay_275__r11": "ccc7d66a7328da454f0fd1cdf732cdb580f16f871562bd2f83c3b99d1a19b641",
      "roleplay_275__r12": "4f2cb099144764f77f0c447bfeb063968e43a0d3af744ab317415f04efb60972",
      "roleplay_275__r13": "a10e160da53c27f2bdafb8bab331554d63cdc8dcb681e79791d3352804ff3ebf",
      "roleplay_275__r14": "3285dc4d26974b561508c61a8071bdcc721ae1af5da08e84e6010b361b60394e",
      "roleplay_275__r15": "589de96112273a560c054236d7339b1b0ef225146572a5d2d0d0657dafe9caf6",
      "roleplay_289__r00": "e2e43afc7169942ac724b315e753a1c9d89e85d7451973b823db849cc0873801",
      "roleplay_289__r01": "fd16f8f5efc4d393d33758cf8bf23070643d8362706b5b5a8f46dbacfc660ebd",
      "roleplay_289__r02": "8c4277d521c4c86b84a6fbd4c8d75ff52547bf815d64264fd8f423262ae8a95a",
      "roleplay_289__r03": "f77494fb96d8adfef60d0bf9042f21e2179eeb31ab6f5109ac7d4a39d40f1661",
      "roleplay_289__r04": "8e8a349a57fc7b21297d4940f17648dbc1419ed6c136d0585fc634cb15ebb395",
      "roleplay_289__r05": "0af4c01c0276249a59061f8060eec14ff5090c1daab0ae9afb21dfd89db173be",
      "roleplay_289__r06": "48e66f5ca0135551f6eed5f49bbad7ce4c7adf5444ca084b3c71f1256b5528b9",
      "roleplay_289__r07": "e547008714eb907f08cb797ca20a3112933cfccc18606deb9c2d4fb31263ea2b",
      "roleplay_289__r08": "10b64541cc6958d8cff846efdce10789b3cee8cd5689215c3bf7581ea28cd311",
      "roleplay_289__r09": "44b1b10dc872815fda608e3f92ae50b4cfad754e47c89f91714826392b5ebb59",
      "roleplay_289__r10": "08381f10176eda7c30e8733f972fc5df0c9dff049587cc9515fd5bc18689a0c9",
      "roleplay_289__r11": "b2f989088f8a970b4356c16af171fbf9f17817ffe125e72e40d2fb4790734c37",
      "roleplay_289__r12": "d7fac6a5a8b5621b2be298cfbcc6de6a883ae2b68eb753f5bf049e5a9ab62a01",
      "roleplay_289__r13": "8c995b88a5061b3f1d711a64ab34b588cdd37e135f06f47dc70042c099215847",
      "roleplay_289__r14": "a38818715aa65c4aaca10c700b4f99dbe016f69669430243327aa32cc2f4b193",
      "roleplay_289__r15": "abda65eb43cc3839be31d7c0475db79b8a04597c27f8f35bce94787380cbb11f",
      "roleplay_297__r00": "ebfe414f6201dd41ce61243bb340c92181410ddfe2b75ed789f857f48610eb91",
      "roleplay_297__r01": "8ff43a4ac1ca10ad54227fb493f54d2ee8295e17a7ac8e070162976e04a9de78",
      "roleplay_297__r02": "386757ef222106d171df16c23f4c9fc450b46527b389f0d981a17a552ac93826",
      "roleplay_297__r03": "f6df07d89cf91b3b9babc9c1274978fff3fffc3f9f1065898388a8725e634744",
      "roleplay_297__r04": "6e6ec42f9ab4eb6e1e65115b2b4d531bc939e5b643cfbbf74f97d973e19a671e",
      "roleplay_297__r05": "d21ac1aba9866b8da4935d7997fe5b102f8cf1bf5bc6c872eed5d23aa578d486",
      "roleplay_297__r06": "7f4ec6064cc5e1c1281b5a8729c8fd16404e96328210fd64a41b456a14e13aac",
      "roleplay_297__r07": "93e68de193ef129f26b4a1f26acb0826a0004c1aebf8a98e20f557d4350b8d93",
      "roleplay_297__r08": "2a4f26a095605956e43c590e0daa55aa225ee1f92e3e230398c29cdc26fe72b9",
      "roleplay_297__r09": "503f4e152f29ec4d7245a3cd70c49c5934e35784b0eee203ec8866a5026f5c5d",
      "roleplay_297__r10": "3d7486d5b52ea8ef12aae0399b9e1497b89b420c649795439825d05399872934",
      "roleplay_297__r11": "40990bc827c1bf68d3d9122ea3628a140493671a2c73641e53097aaf896b4bb5",
      "roleplay_297__r12": "9b323c10569622cccfd2167a5705b3ad4fd95c8dae5a8a0dc524f0631b551fff",
      "roleplay_297__r13": "ec765c07697bd28def55c2363a17cf7682a3f582d36d8b455f98393621c216ba",
      "roleplay_297__r14": "6bcc0005a4f3cb3f5827ce9d9009c8c5273ebc9159c74feb004649c4d689417f",
      "roleplay_297__r15": "1a5f3ac2661789ce85a376feb7dcd32405f2936692c8a2b36f475bcbb7f3002b",
      "roleplay_298__r00": "292d3c344ad9776857b0b9d19b7e48cf682fdf2194cee642d2f9fc8c548696f9",
      "roleplay_298__r01": "0bfa3f333fea46646c4af48abe90232738bc0e3c5013e93da4ecaccfce45c08a",
      "roleplay_298__r02": "228ff668e924187b17260e9b0728e67703b2baceff38488bf41138673925e02f",
      "roleplay_298__r03": "d4698a42ee69947dfc075725258e5212e3524536e6f063274dd0efe8181e62f6",
      "roleplay_298__r04": "5e2bb1603782e030da40591e33deeb355a3e85cc5214b870e23d8bdf88b36536",
      "roleplay_298__r05": "84b1899e363f4755edb5e00bcc0f2af95af1b8a49005e156bbfa998610103fd1",
      "roleplay_298__r06": "4f0cfbee337be447bb465f1da353609425614031e9c43d0ef3a1e027fdbbb337",
      "roleplay_298__r07": "42b083279815fa582bfc1288633731ffe7390aba444bbdbaf18e7bdfe9c3bc4c",
      "roleplay_298__r08": "22be6f2407b581106587d75874c2102745ca469bf5a95b8a941313ca161492f3",
      "roleplay_298__r09": "a304fdc3f5ff73e0fad1f992d9fa1b52067a7f91b5260a1e0c27b0dc11808849",
      "roleplay_298__r10": "0fd67712da58a73f49069219790a0565c5a136f72299090c271d7bd790352e21",
      "roleplay_298__r11": "2d0982dc0a34d1a505dce57944fadaa7637e45d676ae4fe7704f2eb6ac54190c",
      "roleplay_298__r12": "8ca814ccb77229aaa289eb7d1af2b08727ee17f31eac9965e8f7289be3f29eea",
      "roleplay_298__r13": "de2bc5e7b2b526e210e890f8f6d1367398685766ce9ae64d53d68a6005330ac9",
      "roleplay_298__r14": "094c9c1a71ba05978db8ef0c716a12fa6514665d8e54ebacfd187ed18966db81",
      "roleplay_298__r15": "87f565fa1ce4220c3f0cce88c3c30dede347d27ecf83d1534a1918510663882c"
    }
  },
  "n_labels_completed": 176,
  "n_prompts": 11,
  "n_trajectories_completed": 176,
  "n_trajectories_requested": 176,
  "note": "Behavioral outcomes and activation statistics by label are sealed until Phase 24E freezes the candidate (time, layer).",
  "omitted": [
    "H_A_D_totals",
    "per_prompt_H_A_D",
    "mixed_prompt_counts",
    "example_responses",
    "activation_stats_by_label",
    "predictive_performance"
  ],
  "sealed": true,
  "split": "LOCKED_TEST",
  "technical_integrity_ok": true
}
```

Behavioral outcomes for TEST are **not reported**.

## Runtime / cost

- Capture wall seconds: 3321.2750494480133
- Grade wall seconds: 2457.3846101760864
- GPU: A100-80GB
- Estimated cost USD: 4.012958096961181
- Peak memory bytes: 14591429632

## Authorizations after freeze

Live capture / Gemma grading / physiology / TEST scientific analysis: **false**.
