"""
Cargador de Bloques Bitcoin Reales

Parsea bloques_100.csv y los convierte a formato procesable.
"""

import csv
import struct
from typing import List, Dict, Tuple
from pathlib import Path


def load_blocks_csv(csv_path: str) -> List[Dict]:
    """
    Carga bloques desde CSV.
    
    Formato CSV:
    height,hash,merkleroot,tx_count
    
    Nota: El CSV NO contiene el header completo, solo hash y merkleroot.
    Para obtener el header completo necesitaríamos consultar la blockchain.
    
    Por ahora, usaremos estos bloques para validar el sistema con datos reales.
    """
    blocks = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            block = {
                'height': int(row['height']),
                'hash': row['hash'],
                'merkleroot': row['merkleroot'],
                'tx_count': int(row['tx_count'])
            }
            blocks.append(block)
    
    return blocks


def analyze_blocks(blocks: List[Dict]) -> Dict:
    """Analiza estadísticas de los bloques cargados"""
    stats = {
        'total_blocks': len(blocks),
        'height_range': (
            min(b['height'] for b in blocks),
            max(b['height'] for b in blocks)
        ),
        'total_transactions': sum(b['tx_count'] for b in blocks),
        'avg_tx_per_block': sum(b['tx_count'] for b in blocks) / len(blocks) if blocks else 0
    }
    
    return stats


def print_block_stats(stats: Dict):
    """Imprime estadísticas de bloques"""
    print("=" * 80)
    print("  BLOQUES BITCOIN REALES CARGADOS")
    print("=" * 80)
    print(f"\n  Total de bloques:     {stats['total_blocks']}")
    print(f"  Rango de alturas:     {stats['height_range'][0]:,} - {stats['height_range'][1]:,}")
    print(f"  Total transacciones:  {stats['total_transactions']:,}")
    print(f"  Promedio tx/bloque:   {stats['avg_tx_per_block']:.1f}")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    # Cargar bloques
    csv_path = Path(__file__).parent / 'bloques_100.csv'
    blocks = load_blocks_csv(str(csv_path))
    
    # Analizar
    stats = analyze_blocks(blocks)
    print_block_stats(stats)
    
    # Mostrar primeros 5 bloques
    print("Primeros 5 bloques:\n")
    for i, block in enumerate(blocks[:5], 1):
        print(f"{i}. Altura {block['height']:,} - {block['tx_count']} transacciones")
        print(f"   Hash: {block['hash']}")
        print(f"   Merkle: {block['merkleroot']}\n")
