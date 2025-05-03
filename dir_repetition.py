import csv
import ast
import re
import json
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
def process_csv(csv_path):
    """Process CSV with 24 prompts × 10 generations each + header"""
    prompts = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        
        current_prompt = None
        prompt_counter = 0
        
        for i, row in enumerate(reader):
            full_prompt_id = row[0].split('_')[0]
            
            # New prompt group detection
            if not current_prompt or (i % 10 == 0 and i > 0):
                current_prompt = f"prompt_{prompt_counter:02d}"
                prompt_counter += 1
            
            # Store generation with its original metadata
            prompts[current_prompt].append({
                'original_id': full_prompt_id,
                'generation': int(row[1]),
                'ner_raw': row[2],
                'cleaned_ner': row[4]
            })
    
    return prompts

def safe_parse(ner_str):
    """Robust NER parsing with multiple fallbacks"""
    try:
        return json.loads(ner_str.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(ner_str)
        except:
            return {}

def analyze_prompt_group(prompt_group):
    """Analyze 10 generations with proper direct/indirect metrics"""
    direct_tracker = defaultdict(lambda: defaultdict(set))  
    component_tracker = defaultdict(lambda: defaultdict(set))
    
    for gen_num, gen in enumerate(prompt_group):
        ner = safe_parse(gen['cleaned_ner']) or safe_parse(gen['ner_raw'])
        
        for ent_type, entities in ner.items():
            for ent in entities:
                # Direct repetition tracking
                direct_tracker[ent_type][ent].add(gen_num)
                
                # Indirect component tracking
                components = re.findall(r'\w+', ent.lower())
                for comp in components:
                    component_tracker[ent_type][comp].add(gen_num)
    
    direct_metrics = defaultdict(dict)
    component_metrics = defaultdict(dict)
    
    for ent_type in ['PER', 'LOC', 'ORG', 'DATE']:
        total_direct = sum(len(v) for v in direct_tracker[ent_type].values())
        unique_direct = len(direct_tracker[ent_type])
        direct_metrics[ent_type] = {
            'repetition_rate': (total_direct - unique_direct) / total_direct if total_direct else 0,
            'cross_gen_entities': [ent for ent, gens in direct_tracker[ent_type].items() if len(gens) > 1]
        }
        
        total_components = sum(len(v) for v in component_tracker[ent_type].values())
        unique_components = len(component_tracker[ent_type])
        component_metrics[ent_type] = {
            'repetition_rate': (total_components - unique_components) / total_components if total_components else 0,
            'cross_gen_components': [comp for comp, gens in component_tracker[ent_type].items() if len(gens) > 1]
        }
    
    return direct_metrics, component_metrics

def generate_report(prompt_id, direct, component, language):
    """Visualize direct vs indirect repetition rates"""
    output_dir = f"reports/{language}"
    os.makedirs(output_dir, exist_ok=True) 
    
    ent_types = ['PER', 'LOC', 'ORG', 'DATE']
    
    # Data preparation
    direct_rates = [direct[et]['repetition_rate'] for et in ent_types]
    component_rates = [component[et]['repetition_rate'] for et in ent_types]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ent_types))
    width = 0.35
    
    plt.bar(x - width/2, direct_rates, width, label='Direct Entities')
    plt.bar(x + width/2, component_rates, width, label='Components')
    
    plt.title(f'Repetition Rates - {prompt_id} ({language})')
    plt.xticks(x, ent_types)
    plt.ylim(0, 1)
    plt.legend()
    
    # Save to properly created directory
    plt.savefig(f"{output_dir}/{prompt_id}_rates.png")
    plt.close()



def generate_combined_report(all_results, language):
    """Create a single report file with all prompts' stats"""
    output_dir = f"reports/{language}"
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f"{output_dir}/combined_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"=== Combined Repetition Report ({language}) ===\n\n")
        
        for prompt_id, (direct, component) in all_results.items():
            f.write(f"---- {prompt_id} ----\n")
            f.write("| Entity | Direct Rate | Component Rate |\n")
            f.write("|--------|-------------|-----------------|\n")
            
            for et in ['PER', 'LOC', 'ORG', 'DATE']:
                dir_rate = direct.get(et, {}).get('repetition_rate', 0)
                comp_rate = component.get(et, {}).get('repetition_rate', 0)
                f.write(f"| {et:4} | {dir_rate:>10.2%} | {comp_rate:>15.2%} |\n")
            
            f.write("\nTop Repeating Entities:\n")
            for et in ['PER', 'LOC', 'ORG', 'DATE']:
                ents = direct.get(et, {}).get('cross_gen_entities', [])[:3]
                if ents:
                    f.write(f"- {et}: {', '.join(ents)}\n")
            
            f.write("\nTop Repeating Components:\n")
            for et in ['PER', 'LOC', 'ORG', 'DATE']:
                comps = component.get(et, {}).get('cross_gen_components', [])[:3]
                if comps:
                    f.write(f"- {et}: {', '.join(comps)}\n")
            
            f.write("\n" + "="*50 + "\n")

def plot_language_pair(lang1_data, lang2_data, lang1_name, lang2_name):
    """Plot comparison between two languages"""
    plt.figure(figsize=(15, 8))
    markers = ['o', 's'] 
    colors = ['#1f77b4', '#ff7f0e'] 
    
    # Get sorted prompt IDs based on first language
    prompt_ids = sorted(lang1_data['direct'].keys(), 
                       key=lambda x: int(x.split('_')[1]))
    
    for idx, (lang_data, lang_label) in enumerate([(lang1_data, lang1_name),
                                                  (lang2_data, lang2_name)]):
        x = np.arange(len(prompt_ids))
        
        # Direct rates
        y_direct = [lang_data['direct'][pid] for pid in prompt_ids]
        plt.plot(x, y_direct, 
                 label=f'{lang_label} - Direct',
                 color=colors[idx],
                 marker=markers[idx],
                 linestyle='-',
                 linewidth=2)
        
        # Component rates
        y_component = [lang_data['component'][pid] for pid in prompt_ids]
        plt.plot(x, y_component,
                 label=f'{lang_label} - Component',
                 color=colors[idx],
                 marker=markers[idx],
                 linestyle='--',
                 linewidth=2,
                 alpha=0.7)
    
    plt.title(f'Repetition Rates: {lang1_name} vs {lang2_name}', fontsize=14)
    plt.xlabel('Prompt Number', fontsize=12)
    plt.ylabel('Average Repetition Rate', fontsize=12)
    plt.xticks(np.arange(24), [f'P{i+1}' for i in range(24)], rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.15), loc='upper center')
    plt.tight_layout()
    plt.savefig(f'{lang1_name.lower()}_{lang2_name.lower()}_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main entry point for processing all languages and generating pairwise comparisons"""
    language_config = {
        'EN': '/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/Row_Resourced/en_data/cleaned_ner_output.csv',
        'SW': '/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/sw_data/cleaned_ner_output.csv',
        'KIN': '/Users/manzifabriceniyigaba/Desktop/Kinyarwanda/Row_Resourced/kin_data/cleaned_ner_output.csv'
    }
    
    all_language_data = {}
    
    # Process all languages first
    for lang, csv_path in language_config.items():
        prompts = process_csv(csv_path)
        lang_data = {'direct': {}, 'component': {}}
        
        for prompt_id in sorted(prompts.keys(), key=lambda x: int(x.split('_')[1])):
            generations = prompts[prompt_id]
            if len(generations) != 10:
                continue
                
            direct, component = analyze_prompt_group(generations)
            
            # Calculate average rates
            dir_rates = np.mean([direct[et]['repetition_rate'] 
                               for et in ['PER', 'LOC', 'ORG', 'DATE']])
            comp_rates = np.mean([component[et]['repetition_rate']
                                for et in ['PER', 'LOC', 'ORG', 'DATE']])
            
            lang_data['direct'][prompt_id] = dir_rates
            lang_data['component'][prompt_id] = comp_rates
            
            # Generate individual reports
            generate_report(prompt_id, direct, component, lang)
            generate_combined_report({prompt_id: (direct, component)}, lang)
        
        all_language_data[lang] = lang_data
    
    # Generate pairwise comparisons
    plot_language_pair(all_language_data['EN'], all_language_data['KIN'], 'EN', 'KIN')
    plot_language_pair(all_language_data['EN'], all_language_data['SW'], 'EN', 'SW')
    plot_language_pair(all_language_data['SW'], all_language_data['KIN'], 'SW', 'KIN')

if __name__ == "__main__":
    main()
