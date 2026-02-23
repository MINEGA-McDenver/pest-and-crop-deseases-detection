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

  static const String professionalAdvice =
      'For more detailed diagnosis and treatment options, contact your local '
      'agricultural extension officer or visit your nearest RAB center. '
      'This app provides initial guidance only and does not replace '
      'professional advice.';

  static const String healthyCheckup =
      'Even when your crop looks healthy, schedule regular visits from your '
      'sector agronomist for soil testing and nutrition advice.';

  static final Map<String, DiseaseInfo> all = {
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

  static List<String> getHealthyTips(String cropName) {
    final healthyKey = all.keys.firstWhere(
      (k) => k.contains(cropName.toLowerCase()) && k.contains('healthy'),
      orElse: () => '',
    );
    if (healthyKey.isNotEmpty) {
      return all[healthyKey]!.whatToDo;
    }
    return ['Continue monitoring your crop regularly'];
  }

  static List<DiseaseInfo> getForCrop(String cropName) {
    return all.values
        .where((d) => d.cropName.toLowerCase() == cropName.toLowerCase())
        .toList();
  }

  static const List<String> supportedCrops = [
    'Banana',
    'Beans',
    'Maize',
    'Potato',
  ];

  static bool isCropSupported(String cropName) {
    return supportedCrops.any((c) => c.toLowerCase() == cropName.toLowerCase());
  }
}
