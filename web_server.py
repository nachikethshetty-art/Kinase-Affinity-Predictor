#!/usr/bin/env python3
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

TARGET_KINASES = ['ABL1', 'ABL2', 'KIT', 'PDGFRA', 'PDGFRB', 'SRC', 'LYN', 'YES1', 'FYN', 'FRK', 'EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'KDR', 'FLT1', 'FLT4', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'BRAF', 'RAF1', 'MAP2K1', 'RET', 'MET', '4HJO']

KINASE_FAMILIES = {
    'BCR_ABL': ['ABL1', 'ABL2'], 'KIT': ['KIT'], 'PDGFR': ['PDGFRA', 'PDGFRB'],
    'SRC_Family': ['SRC', 'LYN', 'YES1', 'FYN', 'FRK'], 'EGFR': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4'],
    'VEGFR': ['KDR', 'FLT1', 'FLT4'], 'FGFR': ['FGFR1', 'FGFR2', 'FGFR3', 'FGFR4'],
    'Other': ['BRAF', 'RAF1', 'MAP2K1', 'RET', 'MET', '4HJO']
}

def extract_protein_features(pdb_content):
    lines = pdb_content.split('\n')
    resolution = 2.0
    for line in lines:
        if line.startswith('REMARK   2 RESOLUTION'):
            try: resolution = float(line.split()[-2])
            except: pass
    atom_count = sum(1 for line in lines if line.startswith('ATOM'))
    sequence_length = max(atom_count // 10, 200)
    structure_quality = min(1.0, 2.5 / max(resolution, 0.5))
    return {'resolution': resolution, 'sequence_length': sequence_length, 'structure_quality': structure_quality}

def extract_ligand_features(pdb_content):
    lines = pdb_content.split('\n')
    heteroatoms = h_donors = h_acceptors = rotatable_bonds = n_heavy_atoms = 0
    for line in lines:
        if line.startswith('HETATM'):
            element = line[76:78].strip()
            if element and element != 'H':
                n_heavy_atoms += 1
                if element in ['N', 'O', 'S', 'F', 'Cl', 'Br', 'I']:
                    heteroatoms += 1
                    if element == 'N': h_donors += 1
                    if element in ['N', 'O']: h_acceptors += 1
    rotatable_bonds = max(1, n_heavy_atoms // 5)
    estimated_mw = n_heavy_atoms * 10
    return {'n_heavy_atoms': max(1, min(n_heavy_atoms, 100)), 'estimated_mw': max(50, min(estimated_mw, 2000)), 'heteroatoms': min(heteroatoms, 20), 'rotatable_bonds': min(rotatable_bonds, 20), 'h_donors': min(h_donors, 10), 'h_acceptors': min(h_acceptors, 10), 'lipophilicity': estimated_mw / 400.0}

def predict_affinity(kinase, ligand_name, protein_features, ligand_features):
    n_heavy = ligand_features['n_heavy_atoms']
    h_bond = ligand_features['h_donors'] + ligand_features['h_acceptors']
    flex = ligand_features['rotatable_bonds']
    hetero = ligand_features['heteroatoms']
    base = -7.8
    h_contrib = h_bond * -0.23
    if n_heavy < 15: size_c = -1.5
    elif n_heavy < 30: size_c = (n_heavy - 15) * 0.15
    elif n_heavy < 50: size_c = 2.25 - (n_heavy - 30) * 0.015
    else: size_c = 1.95 - (n_heavy - 50) * 0.08
    flex_c = flex * 0.25
    hetero_c = hetero * -0.12
    dg = base + h_contrib + size_c - flex_c + hetero_c
    dg = max(-14, min(-5, dg))
    if dg < -11: interp = 'Extremely strong (Kd < 1 nM)'
    elif dg < -9: interp = 'Very strong (1-100 nM)'
    elif dg < -7: interp = 'Strong (0.1-1 µM)'
    elif dg < -5: interp = 'Moderate (1-10 µM)'
    elif dg < -3: interp = 'Weak (10-100 µM)'
    else: interp = 'Very weak (> 100 µM)'
    return {'status': 'success', 'kinase': kinase, 'ligand_name': ligand_name, 'affinity_score': float(dg), 'interpretation': interp, 'ligand_info': {'molecular_weight': ligand_features['estimated_mw'], 'heavy_atoms': ligand_features['n_heavy_atoms'], 'h_donors': ligand_features['h_donors'], 'h_acceptors': ligand_features['h_acceptors'], 'rotatable_bonds': ligand_features['rotatable_bonds']}, 'protein_info': {'resolution': protein_features['resolution'], 'sequence_length': protein_features['sequence_length'], 'structure_quality': protein_features['structure_quality']}}

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            index_path = Path(__file__).parent / 'index.html'
            with open(index_path, 'rb') as f: self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            kinase, ligand_name = data.get('kinase'), data.get('ligand_name', 'Ligand')
            protein_content, ligand_content = data.get('protein_content', ''), data.get('ligand_content', '')
            if not kinase or not protein_content or not ligand_content:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Missing params'}).encode())
                return
            pf = extract_protein_features(protein_content)
            lf = extract_ligand_features(ligand_content)
            result = predict_affinity(kinase, ligand_name, pf, lf)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args): pass

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), RequestHandler)
    print('Server: http://localhost:8000')
    server.serve_forever()
