
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from cmfo.bitcoin import NonceRestrictor, build_header

def debug_mining():
    print("Iniciando Debug Mineria...")
    try:
        # Genesis Block
        block_genesis = {
            'version': 1,
            'prev_block': bytes(32),
            'merkle_root': bytes.fromhex('4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b'),
            'timestamp': 1231006505,
            'bits': 0x1d00ffff,
            'nonce': 2083236893
        }

        print(f"Merkle Root Hex: {block_genesis['merkle_root'].hex()}")

        header = build_header(
            block_genesis['version'],
            block_genesis['prev_block'],
            block_genesis['merkle_root'],
            block_genesis['timestamp'],
            block_genesis['bits'],
            block_genesis['nonce']
        )
        print(f"Header construido ({len(header)} bytes): {header.hex()[:64]}...")

        # Mode conservative
        restrictor = NonceRestrictor(header, empirical_mode='conservative')
        
        t0 = time.time()
        success, reduced_space, reduction_factor = restrictor.reduce_space()
        dt = time.time() - t0

        print(f"Success: {success}")
        print(f"Reduced Space: {reduced_space}")
        print(f"Reduction Factor: {reduction_factor}")
        print(f"Time: {dt*1000:.2f} ms")

        if not success:
            print("FALLO: AC-3 no convergió o inconsistencia.")
            return False
            
        if reduction_factor < 5.0:
            print(f"FALLO: Factor de reducción insuficiente ({reduction_factor})")
            return False

        in_space = restrictor.is_nonce_in_space(block_genesis['nonce'])
        print(f"Nonce in space: {in_space}")
        
        if not in_space:
             print("FALLO: Nonce real fuera del espacio reducido.")
             # Debug why
             # ...
             return False

        print("DEBUG PASS")
        return True

    except Exception as e:
        print(f"EXCEPT: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if debug_mining():
        sys.exit(0)
    else:
        sys.exit(1)
