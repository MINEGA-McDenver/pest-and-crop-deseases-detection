import '../l10n/app_strings.dart';

class DiseaseInfo {
  final String className;
  final String cropName;
  final String displayName;
  final bool isHealthy;
  final String description;
  final List<String> whatToDo;
  final List<String> prevention;
  final String severity;
  final String iconEmoji;

  const DiseaseInfo({
    required this.className,
    required this.cropName,
    required this.displayName,
    required this.isHealthy,
    required this.description,
    required this.whatToDo,
    required this.prevention,
    required this.severity,
    required this.iconEmoji,
  });

  static String get professionalAdvice =>
      AppStrings.trCode('professionalAdvice');

  static String get healthyCheckup => AppStrings.trCode('healthyCheckup');

  static final Map<String, DiseaseInfo> _enAll = {
    'banana_cordana': const DiseaseInfo(
      className: 'banana_cordana',
      cropName: 'Banana',
      displayName: 'Cordana Leaf Spot',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🍌',
      description:
          'Cordana leaf spot is a fungal disease caused by Cordana musae. '
          'It appears as brown oval spots with yellow halos, mainly on older '
          'leaves. It reduces photosynthesis and can weaken the plant over time.',
      whatToDo: [
        'Remove affected leaves and burn them',
        'Apply copper-based fungicide (copper oxychloride)',
        'Improve air circulation between plants by pruning',
        'Avoid wetting leaves during irrigation',
      ],
      prevention: [
        'Space plants adequately (3m × 3m)',
        'Remove dead and fallen leaves regularly',
        'Avoid wounding plants during cultivation',
        'Use disease-free planting material',
        'Ensure good drainage in the plantation',
      ],
    ),
    'banana_pestalotiopsis': const DiseaseInfo(
      className: 'banana_pestalotiopsis',
      cropName: 'Banana',
      displayName: 'Pestalotiopsis',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🍌',
      description:
          'Pestalotiopsis is a fungal disease causing brown to black spots '
          'with concentric rings on banana leaves. It thrives in humid '
          'conditions and can spread rapidly in dense plantations.',
      whatToDo: [
        'Remove and destroy all infected leaves',
        'Apply systemic fungicide (Carbendazim)',
        'Reduce humidity around plants by pruning suckers',
        'Avoid overhead irrigation',
      ],
      prevention: [
        'Ensure good air circulation in the plantation',
        'Use disease-free planting material',
        'Maintain proper plant spacing',
        'Remove plant debris from the field',
        'Apply preventive fungicide during rainy season',
      ],
    ),
    'banana_sigatoka': const DiseaseInfo(
      className: 'banana_sigatoka',
      cropName: 'Banana',
      displayName: 'Sigatoka Leaf Spot',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🍌',
      description:
          'Sigatoka (Black/Yellow) is a major fungal disease of banana. It causes '
          'yellow streaks that turn brown or black, severely reducing '
          'photosynthesis and fruit yield. It can destroy up to 50% of leaf area.',
      whatToDo: [
        'Apply fungicide (Propiconazole or Chlorothalonil) on a 2-week cycle',
        'Remove heavily infected leaves immediately',
        'De-leaf regularly — remove oldest leaves showing symptoms',
        'Monitor closely during rainy season',
      ],
      prevention: [
        'Use resistant varieties (FHIA-17, FHIA-23)',
        'Maintain proper spacing (3m × 3m)',
        'Remove excess suckers to reduce plant density',
        'De-leaf regularly to remove old leaves',
        'Apply preventive fungicide before symptoms appear',
        'Improve drainage to reduce humidity',
      ],
    ),
    'banana_healthy': const DiseaseInfo(
      className: 'banana_healthy',
      cropName: 'Banana',
      displayName: 'Healthy',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🍌',
      description:
          'Your banana plant looks healthy! No visible signs of disease detected.',
      whatToDo: [
        'Continue current management practices',
        'Monitor leaves weekly for early signs of disease',
        'Apply potassium-rich fertilizer for strong growth',
        'Mulch around the base to retain moisture',
        'Remove old and dry leaves regularly',
      ],
      prevention: [
        'Maintain good plantation hygiene',
        'Use clean planting material',
        'Ensure proper drainage',
        'Inspect plants regularly, especially during rainy season',
      ],
    ),
    'beans_angular_leaf_spot': const DiseaseInfo(
      className: 'beans_angular_leaf_spot',
      cropName: 'Beans',
      displayName: 'Angular Leaf Spot',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🫘',
      description:
          'Angular leaf spot is a bacterial disease caused by Pseudomonas '
          'syringae. It shows as angular brown spots bounded by leaf veins. '
          'It spreads rapidly in wet conditions and can reduce yield by 40-80%.',
      whatToDo: [
        'Remove and destroy infected plants',
        'Apply copper hydroxide spray',
        'Do NOT work in the field when plants are wet (spreads bacteria)',
        'Harvest early if infection is severe',
      ],
      prevention: [
        'Use certified disease-free seeds',
        'Rotate crops on a 2-3 year cycle',
        'Avoid overhead watering',
        'Plant resistant bean varieties',
        'Ensure adequate spacing for air circulation',
      ],
    ),
    'beans_rust': const DiseaseInfo(
      className: 'beans_rust',
      cropName: 'Beans',
      displayName: 'Bean Rust',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🫘',
      description:
          'Bean rust is a fungal disease caused by Uromyces appendiculatus. '
          'It appears as small reddish-brown pustules on the undersides of '
          'leaves. Severe infection can cause defoliation and yield loss.',
      whatToDo: [
        'Apply fungicide (Mancozeb or Triazole-based)',
        'Remove severely affected plants and burn them',
        'Harvest early if infection is heavy',
        'Avoid working in wet fields to prevent spread',
      ],
      prevention: [
        'Plant resistant bean varieties',
        'Avoid planting beans in the same field consecutively',
        'Ensure adequate spacing for air circulation',
        'Remove crop residue after harvest',
        'Plant early in the season',
      ],
    ),
    'beans_healthy': const DiseaseInfo(
      className: 'beans_healthy',
      cropName: 'Beans',
      displayName: 'Healthy',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🫘',
      description:
          'Your bean plant looks healthy! No visible signs of disease detected.',
      whatToDo: [
        'Continue monitoring regularly',
        'Apply balanced NPK fertilizer',
        'Control weeds around plants',
        'Water consistently but avoid waterlogging',
      ],
      prevention: [
        'Rotate with non-legume crops',
        'Use certified seeds each season',
        'Maintain field hygiene',
        'Scout weekly for early signs of disease',
      ],
    ),
    'maize_common_rust': const DiseaseInfo(
      className: 'maize_common_rust',
      cropName: 'Maize',
      displayName: 'Common Rust',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🌽',
      description:
          'Common rust is a fungal disease caused by Puccinia sorghi. It appears '
          'as small, circular to elongate brown or red pustules on both leaf '
          'surfaces. Severe infections reduce grain filling and yield.',
      whatToDo: [
        'Apply fungicide (Mancozeb or Azoxystrobin) if infection is early',
        'Remove heavily infected lower leaves',
        'Ensure adequate nutrition to help plants resist',
        'Monitor spread — treat neighboring plants preventively',
      ],
      prevention: [
        'Plant resistant hybrid varieties',
        'Plant early in the season to avoid peak infection period',
        'Avoid late planting',
        'Ensure proper plant spacing',
        'Remove volunteer maize plants',
      ],
    ),
    'maize_gray_leaf_spot': const DiseaseInfo(
      className: 'maize_gray_leaf_spot',
      cropName: 'Maize',
      displayName: 'Gray Leaf Spot',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🌽',
      description:
          'Gray leaf spot is a fungal disease caused by Cercospora zeae-maydis. '
          'It shows as long, rectangular gray or tan lesions running parallel '
          'to leaf veins. It is one of the most damaging maize diseases globally.',
      whatToDo: [
        'Apply foliar fungicide (Strobilurin-based)',
        'Remove crop residue after harvest',
        'Ensure adequate plant nutrition',
        'Scout frequently — early treatment is critical',
      ],
      prevention: [
        'Rotate with non-cereal crops for at least one season',
        'Till crop residue into soil after harvest',
        'Use resistant maize varieties',
        'Avoid continuous maize cultivation on the same field',
        'Maintain proper plant density',
      ],
    ),
    'maize_northern_leaf_blight': const DiseaseInfo(
      className: 'maize_northern_leaf_blight',
      cropName: 'Maize',
      displayName: 'Northern Leaf Blight',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🌽',
      description:
          'Northern leaf blight is caused by the fungus Exserohilum turcicum. '
          'It produces long, elliptical gray-green lesions (2-15 cm) on leaves. '
          'Severe cases can cause significant yield loss.',
      whatToDo: [
        'Apply fungicide at first sign of symptoms',
        'Remove lower infected leaves',
        'Ensure good air circulation in the field',
        'Apply balanced fertilizer to strengthen plants',
      ],
      prevention: [
        'Use resistant maize varieties',
        'Rotate crops — avoid maize after maize',
        'Destroy crop residue after harvest',
        'Plant at recommended density',
        'Avoid planting in poorly drained areas',
      ],
    ),
    'maize_healthy': const DiseaseInfo(
      className: 'maize_healthy',
      cropName: 'Maize',
      displayName: 'Healthy',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🌽',
      description:
          'Your maize plant looks healthy! No visible signs of disease detected.',
      whatToDo: [
        'Continue monitoring weekly',
        'Apply nitrogen fertilizer at knee-high stage',
        'Ensure consistent watering',
        'Control weeds to reduce competition',
      ],
      prevention: [
        'Ensure proper spacing between plants',
        'Scout weekly during the rainy season',
        'Maintain good field sanitation',
        'Apply fertilizer according to soil test results',
      ],
    ),
    'potato_early_blight': const DiseaseInfo(
      className: 'potato_early_blight',
      cropName: 'Potato',
      displayName: 'Early Blight',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🥔',
      description:
          'Early blight is caused by the fungus Alternaria solani. It shows as '
          'dark brown spots with concentric rings (target-like pattern) on older '
          'leaves first. It gradually spreads upward and reduces tuber size.',
      whatToDo: [
        'Apply fungicide (Chlorothalonil or Mancozeb) every 7-10 days',
        'Remove infected lower leaves',
        'Ensure adequate plant nutrition (potassium helps resistance)',
        'Water at the base — avoid wetting leaves',
      ],
      prevention: [
        'Use certified seed potatoes',
        'Rotate with non-solanaceous crops (3+ years)',
        'Mulch to prevent soil splash onto leaves',
        'Maintain adequate plant spacing',
        'Remove crop debris after harvest',
      ],
    ),
    'potato_late_blight': const DiseaseInfo(
      className: 'potato_late_blight',
      cropName: 'Potato',
      displayName: 'Late Blight',
      isHealthy: false,
      severity: 'critical',
      iconEmoji: '🥔',
      description:
          'Late blight is caused by Phytophthora infestans. It produces large, '
          'irregular water-soaked lesions with white mold on leaf undersides. '
          'THIS IS EXTREMELY DESTRUCTIVE — it can destroy an entire field in days.',
      whatToDo: [
        'ACT IMMEDIATELY — do not wait',
        'Apply Metalaxyl or Dimethomorph fungicide urgently',
        'Remove and BURN all infected plants (do NOT compost)',
        'Alert neighboring farmers — this spreads fast',
        'Check tubers — infected tubers rot in storage',
      ],
      prevention: [
        'Use resistant varieties (e.g., Kinigi, Kirundo for Rwanda)',
        'Apply preventive fungicide before and during rainy season',
        'Avoid overhead irrigation',
        'Hill soil around stems to protect tubers',
        'Monitor weather — humid/cool conditions favor outbreaks',
        'Never plant near last season\'s potato field',
      ],
    ),
    'potato_healthy': const DiseaseInfo(
      className: 'potato_healthy',
      cropName: 'Potato',
      displayName: 'Healthy',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🥔',
      description:
          'Your potato plant looks healthy! No visible signs of disease detected.',
      whatToDo: [
        'Continue hilling soil around stems',
        'Monitor leaves regularly, especially during rainy season',
        'Apply calcium and phosphorus fertilizer',
        'Water consistently but avoid waterlogging',
      ],
      prevention: [
        'Hill soil 2-3 times during the growth cycle',
        'Monitor especially during humid/rainy periods',
        'Remove volunteer potato plants from previous season',
        'Maintain good field drainage',
      ],
    ),
  };

  static final Map<String, DiseaseInfo> _rwAll = {
    'banana_cordana': const DiseaseInfo(
      className: 'banana_cordana',
      cropName: 'Banana',
      displayName: 'Cordana Leaf Spot',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🍌',
      description:
          'Cordana ni indwara y\'ubusazi iterwa na Cordana musae. '
          'Igaragara nk\'udutâches twijimye tuzunguruka dufite impande z\'umuhondo, '
          'cyane ku mababi ashaje. Igabanya photosynthesis kandi igatuma igihingwa kigabanuka imbaraga uko iminsi igenda.',
      whatToDo: [
        'Kura amababi arwaye kandi uyatwike',
        'Shyiraho umuti urimo copper (copper oxychloride)',
        'Ongera uburyo umwuka unyura hagati y\'ibimera ukoresheje gutema',
        'Irinde gutosa amababi igihe uhira',
      ],
      prevention: [
        'Tera usize intera ihagije (3m × 3m)',
        'Kura amababi yumye n\'aguye hasi buri gihe',
        'Irinde gukomeretsa ibimera igihe ukora mu murima',
        'Koresha ingemwe zidafite indwara',
        'Emeza ko amazi ataguma mu murima',
      ],
    ),
    'banana_pestalotiopsis': const DiseaseInfo(
      className: 'banana_pestalotiopsis',
      cropName: 'Banana',
      displayName: 'Pestalotiopsis Leaf Spot',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🍌',
      description:
          'Pestalotiopsis ni indwara y\'ubusazi itera udutâches twijimye kugeza ku mwirabura '
          'dufite impeta hagati ku mababi y\'urutoki. Ikunda ahantu hafite ubuhehere bwinshi '
          'kandi ikwirakwira vuba iyo urutoki ruteranye cyane.',
      whatToDo: [
        'Kura amababi yose arwaye kandi uyatwike',
        'Koresha umuti winjira mu gihingwa (Carbendazim)',
        'Gabanya ubuhehere ukoresheje kugabanya insina nyinshi',
        'Irinde kuhira hejuru y\'amababi',
      ],
      prevention: [
        'Emeza ko umwuka unyura neza mu rutoki',
        'Koresha ingemwe zidafite indwara',
        'Hubahiriza intera yo gutera',
        'Kura ibisigazwa by\'ibimera mu murima',
        'Koresha umuti wo kwirinda cyane mu gihe cy\'imvura',
      ],
    ),
    'banana_sigatoka': const DiseaseInfo(
      className: 'banana_sigatoka',
      cropName: 'Banana',
      displayName: 'Sigatoka Leaf Spot',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🍌',
      description:
          'Sigatoka (umukara cyangwa umuhondo) ni indwara ikomeye y\'urutoki. '
          'Ibanza nk\'imirongo y\'umuhondo ikaza kuba ikijimye cyangwa umukara, '
          'igabanya cyane photosynthesis n\'umusaruro. Ishobora kwangiza kugera kuri 50% by\'ikibabi.',
      whatToDo: [
        'Shyiraho umuti (Propiconazole cyangwa Chlorothalonil) buri byumweru 2',
        'Kura amababi arwaye cyane ako kanya',
        'Komeza gukuraho amababi ashaje agaragaza ibimenyetso',
        'Kurikiranira hafi cyane mu gihe cy\'imvura',
      ],
      prevention: [
        'Koresha ubwoko bwihanganira indwara (FHIA-17, FHIA-23)',
        'Hubahiriza intera yo gutera (3m × 3m)',
        'Gabanya insina nyinshi kugira ngo urutoki rutaterana cyane',
        'Kuraho amababi ashaje buri gihe',
        'Shyira umuti wo kwirinda mbere y\'uko ibimenyetso bigaragara',
        'Tunganya imiyoboro y\'amazi kugira ngo ubuhehere bugabanuke',
      ],
    ),
    'banana_healthy': const DiseaseInfo(
      className: 'banana_healthy',
      cropName: 'Banana',
      displayName: 'Gifite ubuzima bwiza',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🍌',
      description:
          'Umutoki wawe usa n\'ufite ubuzima bwiza! Nta bimenyetso by\'indwara bigaragara.',
      whatToDo: [
        'Komeza uburyo bwiza usanzwe ukoresha',
        'Genzura amababi buri cyumweru kugira ngo ubone ibimenyetso hakiri kare',
        'Koresha ifumbire ifite potasiyumu ihagije',
        'Shyira ibisigazwa by\'ibimera hasi (mulch) ngo amazi agume mu butaka',
        'Kura amababi ashaje kandi yumye buri gihe',
      ],
      prevention: [
        'Bungabunga isuku y\'urutoki',
        'Koresha ingemwe zisukuye',
        'Emeza ko amazi atindaho',
        'Suzuma ibimera kenshi cyane cyane mu gihe cy\'imvura',
      ],
    ),
    'beans_angular_leaf_spot': const DiseaseInfo(
      className: 'beans_angular_leaf_spot',
      cropName: 'Beans',
      displayName: 'Angular Leaf Spot',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🫘',
      description:
          'Iyi ndwara iterwa na bakteriya Pseudomonas syringae. '
          'Igaragara nk\'udutâches twijimye dufungwa n\'imitsi y\'ikibabi. '
          'Ikwirakwira vuba mu buhehere kandi ishobora kugabanya umusaruro hagati ya 40-80%.',
      whatToDo: [
        'Kuraho ibihingwa byanduye kandi ubisenye cyangwa ubisatire',
        'Koresha umuti wa copper hydroxide',
        'Ntukore mu murima ibimera bikiri bitose (bikwirakwiza bakteriya)',
        'Niba indwara ikomeye, sarura hakiri kare',
      ],
      prevention: [
        'Koresha imbuto zemewe kandi zidafite indwara',
        'Hinduranya ibihingwa buri myaka 2-3',
        'Irinde kuhira hejuru y\'ibimera',
        'Tera ubwoko bw\'ibishyimbo bwihanganira indwara',
        'Tera usize intera ihagije ngo umwuka unyure',
      ],
    ),
    'beans_rust': const DiseaseInfo(
      className: 'beans_rust',
      cropName: 'Beans',
      displayName: 'Bean Rust',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🫘',
      description:
          'Rusti y\'ibishyimbo ni indwara y\'ubusazi iterwa na Uromyces appendiculatus. '
          'Igaragara nk\'ududomo tw\'umutuku wijimye ku ruhande rwo hasi rw\'ikibabi. '
          'Iyo ikabije ishobora gutuma amababi ahunguka no kugabanya umusaruro.',
      whatToDo: [
        'Koresha umuti (Mancozeb cyangwa Triazole)',
        'Kuraho ibimera byafashwe cyane kandi ubisatire',
        'Niba byakabije, sarura hakiri kare',
        'Irinde gukora mu murima utose kugira ngo idakwirakwira',
      ],
      prevention: [
        'Tera ubwoko bwihanganira rusti',
        'Irinde guhinga ibishyimbo ahantu hamwe buri gihe',
        'Hubahiriza intera ihagije',
        'Kuraho ibisigazwa by\'ibihingwa nyuma yo gusarura',
        'Tera hakiri mu gihembwe',
      ],
    ),
    'beans_healthy': const DiseaseInfo(
      className: 'beans_healthy',
      cropName: 'Beans',
      displayName: 'Gifite ubuzima bwiza',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🫘',
      description:
          'Igihingwa cyawe cy\'ibishyimbo kimeze neza! Nta bimenyetso by\'indwara bigaragara.',
      whatToDo: [
        'Komeza gukurikirana buri gihe',
        'Koresha ifumbire iboneye ya NPK',
        'Rwanya urumamfu ruzengurutse ibimera',
        'Hira neza ariko wirinde ko amazi adindira',
      ],
      prevention: [
        'Hinduranya n\'ibihingwa bitari ibinyamisogwe',
        'Koresha imbuto zemewe buri gihembwe',
        'Bungabunga isuku y\'umurima',
        'Genzura buri cyumweru ibimenyetso by\'indwara',
      ],
    ),
    'maize_common_rust': const DiseaseInfo(
      className: 'maize_common_rust',
      cropName: 'Maize',
      displayName: 'Common Rust',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🌽',
      description:
          'Rusti isanzwe y\'ibigori iterwa na Puccinia sorghi. '
          'Igaragara nk\'ududomo duto twijimye cyangwa dutukura ku mpande zombi z\'ikibabi. '
          'Iyo ikabije igabanya kwuzura kw\'ibigori n\'umusaruro.',
      whatToDo: [
        'Koresha umuti (Mancozeb cyangwa Azoxystrobin) niba indwara itangiye hakiri kare',
        'Kuraho amababi yo hasi yanduye cyane',
        'Shyiramo intungamubiri zihagije ngo igihingwa kigire imbaraga',
        'Kurikiranira hafi kandi urinde ibimera byo hafi',
      ],
      prevention: [
        'Tera ubwoko bw\'ibigori bwihanganira indwara',
        'Tera hakiri kare kugirango wirinde igihe indwara ikunda',
        'Irinde gutera bitinze',
        'Hubahiriza intera yo gutera',
        'Kuraho ibigori byimeze byonyine byamaze gusigara mu murima',
      ],
    ),
    'maize_gray_leaf_spot': const DiseaseInfo(
      className: 'maize_gray_leaf_spot',
      cropName: 'Maize',
      displayName: 'Gray Leaf Spot',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🌽',
      description:
          'Iyi ndwara iterwa na Cercospora zeae-maydis. '
          'Igaragara nk\'udutâches dutureture dufite ibara ry\'imvi cyangwa ikigina, '
          'dukurikira imitsi y\'ikibabi. Ni imwe mu ndwara zikomeye cyane z\'ibigori.',
      whatToDo: [
        'Koresha umuti wo ku mababi (strobilurin-based)',
        'Kura ibisigazwa by\'ibihingwa nyuma yo gusarura',
        'Shyiramo ifumbire iboneye',
        'Genzura kenshi kuko kwivuza hakiri kare ari ingenzi',
      ],
      prevention: [
        'Hinduranya n\'ibihingwa bitari ibinyampeke nibura igihembwe kimwe',
        'Shyira ibisigazwa by\'ibihingwa mu butaka nyuma yo gusarura',
        'Koresha ubwoko bw\'ibigori bwihanganira indwara',
        'Irinde guhinga ibigori ahantu hamwe buri gihe',
        'Hubahiriza ubwinshi bukwiye bw\'ibimera',
      ],
    ),
    'maize_northern_leaf_blight': const DiseaseInfo(
      className: 'maize_northern_leaf_blight',
      cropName: 'Maize',
      displayName: 'Northern Leaf Blight',
      isHealthy: false,
      severity: 'high',
      iconEmoji: '🌽',
      description:
          'Iyi ndwara iterwa n\'ubusazi Exserohilum turcicum. '
          'Igaragaza udutâches dutureture dufite ibara ry\'imvi-icyatsi (2-15 cm) ku mababi. '
          'Iyo ikabije ishobora gutuma umusaruro ugabanuka cyane.',
      whatToDo: [
        'Shyiraho umuti ukimara kubona ibimenyetso bya mbere',
        'Kuraho amababi yo hasi yanduye',
        'Emeza ko umwuka unyura neza mu murima',
        'Koresha ifumbire iboneye kugirango ibihingwa bikomere',
      ],
      prevention: [
        'Koresha ubwoko bw\'ibigori bwihanganira indwara',
        'Hinduranya ibihingwa, wirinde ibigori nyuma y\'ibigori',
        'Senya ibisigazwa by\'ibihingwa nyuma yo gusarura',
        'Tera ku ntera isabwa',
        'Irinde gutera ahantu amazi adatembera neza',
      ],
    ),
    'maize_healthy': const DiseaseInfo(
      className: 'maize_healthy',
      cropName: 'Maize',
      displayName: 'Gifite ubuzima bwiza',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🌽',
      description:
          'Igihingwa cyawe cy\'ibigori kimeze neza! Nta bimenyetso by\'indwara bigaragara.',
      whatToDo: [
        'Komeza kugenzura buri cyumweru',
        'Shyira ifumbire ya azote ku gihe (igihe ibigori bigeze ku ivi)',
        'Hira neza kandi buri gihe',
        'Rwanya urumamfu kugirango rutarushanwa n\'igihingwa',
      ],
      prevention: [
        'Hubahiriza intera iboneye hagati y\'ibimera',
        'Genzura buri cyumweru cyane cyane mu gihe cy\'imvura',
        'Bungabunga isuku y\'umurima',
        'Koresha ifumbire hashingiwe ku bipimo by\'ubutaka',
      ],
    ),
    'potato_early_blight': const DiseaseInfo(
      className: 'potato_early_blight',
      cropName: 'Potato',
      displayName: 'Early Blight',
      isHealthy: false,
      severity: 'medium',
      iconEmoji: '🥔',
      description:
          'Early blight iterwa na Alternaria solani. '
          'Igaragara nk\'udutâches twijimye dufite impeta (nk\'intego) ku mababi ashaje mbere. '
          'Ikwirakwira hejuru y\'igihingwa kandi ikagabanya ubunini bw\'ibirayi.',
      whatToDo: [
        'Koresha umuti (Chlorothalonil cyangwa Mancozeb) buri minsi 7-10',
        'Kuraho amababi yo hasi yanduye',
        'Shyiramo intungamubiri zihagije (potasiyumu ifasha kurwanya indwara)',
        'Hira ku mizi, wirinde gutosa amababi',
      ],
      prevention: [
        'Koresha imbuto z\'ibirayi zemewe',
        'Hinduranya n\'ibihingwa bitari mu muryango wa solanaceae (imyaka 3+)',
        'Shyira mulch kugira ngo ubutaka butagwa ku mababi',
        'Hubahiriza intera iboneye',
        'Kuraho ibisigazwa by\'ibihingwa nyuma yo gusarura',
      ],
    ),
    'potato_late_blight': const DiseaseInfo(
      className: 'potato_late_blight',
      cropName: 'Potato',
      displayName: 'Late Blight',
      isHealthy: false,
      severity: 'critical',
      iconEmoji: '🥔',
      description:
          'Late blight iterwa na Phytophthora infestans. '
          'Igaragara nk\'udutâches tunini tumeze nk\'utwuzuye amazi, dufite umweru ku ruhande rwo hasi rw\'ikibabi. '
          'NI INDWARA IKOMEYE CYANE - ishobora kwangiza umurima wose mu minsi mike.',
      whatToDo: [
        'KORA UBU NYINE - ntutinde',
        'Shyiraho umuti wa Metalaxyl cyangwa Dimethomorph byihutirwa',
        'Kura kandi utwike ibimera byanduye byose (ntubyongere muri compost)',
        'Menyesha abahinzi b\'abaturanyi kuko ikwirakwira vuba',
        'Genzura ibirayi byo hasi kuko byanduye biborera mu bubiko',
      ],
      prevention: [
        'Koresha ubwoko bwihanganira indwara (nka Kinigi, Kirundo mu Rwanda)',
        'Koresha umuti wo kwirinda mbere no mu gihe cy\'imvura',
        'Irinde kuhira hejuru y\'ibimera',
        'Zamura ubutaka ku mizi kugirango urinde ibirayi',
        'Kurikira iteganyagihe - ubuhehere n\'ubukonje byongera iyi ndwara',
        'Ntutere hafi y\'ahari haterwaga ibirayi umwaka ushize',
      ],
    ),
    'potato_healthy': const DiseaseInfo(
      className: 'potato_healthy',
      cropName: 'Potato',
      displayName: 'Gifite ubuzima bwiza',
      isHealthy: true,
      severity: 'low',
      iconEmoji: '🥔',
      description:
          'Igihingwa cyawe cy\'ibirayi kimeze neza! Nta bimenyetso by\'indwara bigaragara.',
      whatToDo: [
        'Komeza kuzamura ubutaka ku mizi y\'igihingwa',
        'Genzura amababi kenshi cyane cyane mu gihe cy\'imvura',
        'Koresha ifumbire irimo calcium na phosphorus',
        'Hira neza ariko wirinde amazi menshi adindira',
      ],
      prevention: [
        'Zamura ubutaka inshuro 2-3 mu gihe cyo gukura',
        'Kurikiranira hafi cyane mu bihe by\'ubuhehere n\'imvura',
        'Kuraho ibirayi byasigaye byongera kumera mu murima',
        'Emeza ko umurima ufite drainage nziza',
      ],
    ),
  };

  static Map<String, DiseaseInfo> get all {
    return AppStrings.activeLanguageCode == 'rw' ? _rwAll : _enAll;
  }

  // Backward-compatibility aliases for older translated labels in saved history.
  static const Map<String, String> _legacyDiseaseNameAliases = {
    'tâches de cordana': 'banana_cordana',
    'pestalotiopsis': 'banana_pestalotiopsis',
    'tâches ya sigatoka': 'banana_sigatoka',
    'tâches z\'impande ku bibabi': 'beans_angular_leaf_spot',
    'rusti y\'ibishyimbo': 'beans_rust',
    'rusti isanzwe y\'ibigori': 'maize_common_rust',
    'tâches imvi ku bibabi': 'maize_gray_leaf_spot',
  };

  static const Map<String, String> _legacyCropAliases = {
    'banana': 'banana',
    'igitoki': 'banana',
    'beans': 'beans',
    'ibishyimbo': 'beans',
    'maize': 'maize',
    'ibigori': 'maize',
    'potato': 'potato',
    'ibirayi': 'potato',
    'unknown': 'unknown',
    'unknown crop': 'unknown',
    'unknowncrop': 'unknown',
    'ntibizwi': 'unknown',
    'igihingwa kitazwi': 'unknown',
  };

  static const Map<String, String> _diagnosisAliases = {
    'healthy': 'healthy',
    'cyiza': 'healthy',
    'uncertain': 'uncertain',
    'ntibirasobanuka': 'uncertain',
    'unknown condition': 'unknownCondition',
    'indwara itazwi': 'unknownCondition',
  };

  static String canonicalCropKey(String cropName) {
    final normalized = cropName.trim().toLowerCase();
    if (normalized.isEmpty) return '';

    final direct = _legacyCropAliases[normalized];
    if (direct != null) return direct;

    // Stored class keys like `banana_sigatoka` should resolve to `banana`.
    final parts = normalized.split('_');
    if (parts.isNotEmpty) {
      final first = _legacyCropAliases[parts.first];
      if (first != null) return first;
    }

    return normalized;
  }

  static String? canonicalDiseaseKey(String diseaseName, {String? classKey}) {
    final normalized = diseaseName.trim().toLowerCase();
    if (normalized.isEmpty) return null;

    if (classKey != null) {
      final normalizedClassKey = classKey.trim().toLowerCase();
      if (_enAll.containsKey(normalizedClassKey)) return normalizedClassKey;
    }

    if (_enAll.containsKey(normalized)) return normalized;

    final legacy = _legacyDiseaseNameAliases[normalized];
    if (legacy != null) return legacy;

    return _diagnosisAliases[normalized];
  }

  static String localizeDiseaseName(String diseaseName, {String? classKey}) {
    final normalized = diseaseName.trim().toLowerCase();
    if (normalized.isEmpty) return diseaseName;

    final canonicalKey = canonicalDiseaseKey(diseaseName, classKey: classKey);
    if (canonicalKey != null) {
      final byClass = all[canonicalKey];
      if (byClass != null) return byClass.displayName;

      if (canonicalKey == 'healthy' ||
          canonicalKey == 'uncertain' ||
          canonicalKey == 'unknownCondition') {
        return AppStrings.trCode(canonicalKey);
      }
    }

    // Match against either language dataset and return current-locale label.
    for (final entry in _enAll.entries) {
      final enName = entry.value.displayName.toLowerCase();
      final rwName = _rwAll[entry.key]?.displayName.toLowerCase();
      if (normalized == enName || normalized == rwName) {
        return all[entry.key]?.displayName ?? diseaseName;
      }
    }

    return diseaseName;
  }

  static String englishDiseaseName(String diseaseName, {String? classKey}) {
    final normalized = diseaseName.trim().toLowerCase();
    if (normalized.isEmpty) return diseaseName;

    final canonicalKey = canonicalDiseaseKey(diseaseName, classKey: classKey);
    if (canonicalKey != null) {
      final byClass = _enAll[canonicalKey];
      if (byClass != null) return byClass.displayName;

      if (canonicalKey == 'healthy') return 'Healthy';
      if (canonicalKey == 'uncertain') return 'Uncertain';
      if (canonicalKey == 'unknownCondition') return 'Unknown Condition';
    }

    for (final entry in _enAll.entries) {
      final enName = entry.value.displayName.toLowerCase();
      final rwName = _rwAll[entry.key]?.displayName.toLowerCase();
      if (normalized == enName || normalized == rwName) {
        return entry.value.displayName;
      }
    }

    return diseaseName;
  }

  static DiseaseInfo? resolveByCropAndDiseaseName(
    String cropName,
    String diseaseName,
  ) {
    final normalizedCrop = canonicalCropKey(cropName);
    final normalizedDisease = diseaseName.trim().toLowerCase();
    if (normalizedCrop.isEmpty || normalizedDisease.isEmpty) return null;

    final canonicalDisease = canonicalDiseaseKey(diseaseName);
    if (canonicalDisease != null) {
      final byClass = _enAll[canonicalDisease];
      if (byClass != null &&
          canonicalCropKey(byClass.cropName) == normalizedCrop) {
        return all[canonicalDisease];
      }
    }

    for (final entry in _enAll.entries) {
      final en = entry.value;
      final rw = _rwAll[entry.key];
      final sameCrop = canonicalCropKey(en.cropName) == normalizedCrop;
      if (!sameCrop) continue;

      final enName = en.displayName.toLowerCase();
      final rwName = rw?.displayName.toLowerCase();
      if (normalizedDisease == enName || normalizedDisease == rwName) {
        return all[entry.key];
      }
    }

    return null;
  }

  static List<String> getHealthyTips(String cropName) {
    final cropKey = canonicalCropKey(cropName);
    final healthyKey = all.keys.firstWhere(
      (k) => k.contains(cropKey) && k.contains('healthy'),
      orElse: () => '',
    );
    if (healthyKey.isNotEmpty) {
      return all[healthyKey]!.whatToDo;
    }
    return [AppStrings.trCode('continueMonitoringCrop')];
  }

  static List<DiseaseInfo> getForCrop(String cropName) {
    final normalized = canonicalCropKey(cropName);
    return all.values
      .where((d) => canonicalCropKey(d.cropName) == normalized)
        .toList();
  }

  static const List<String> supportedCrops = [
    'Banana',
    'Beans',
    'Maize',
    'Potato',
  ];

  static bool isCropSupported(String cropName) {
    final normalized = canonicalCropKey(cropName);
    return supportedCrops
        .map((c) => canonicalCropKey(c))
        .any((c) => c == normalized);
  }
}
