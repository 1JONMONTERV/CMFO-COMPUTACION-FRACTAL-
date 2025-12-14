/*
 * CMFO-UNIVERSE Core
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 *
 * Licencia:
 *   CMFO-UNIVERSE LICENSE v1.1 — Commercial Restricted.
 */

#ifndef CMFO_CORE_H
#define CMFO_CORE_H

// =====================================================
//  CONSTANTES FRACTALES
// =====================================================
extern const double CMFO_PHI;
extern const double CMFO_TAU;

// Acceso como función (para compatibilidad con tests)
double cmfo_phi(void);
double cmfo_tau(void);

// =====================================================
//  VECTORES T7
// =====================================================
double cmfo_norm7(const double a[7]);
double cmfo_dot7(const double a[7], const double b[7]);
void   cmfo_add7(double out[7], const double a[7], const double b[7]);
void   cmfo_scale7(double out[7], const double a[7], double k);

// =====================================================
//  MATRICES 7×7
// =====================================================
void cmfo_mat7_identity(double M[7][7]);
void cmfo_mat7_zero(double M[7][7]);
void cmfo_mat7_transpose(double out[7][7], const double M[7][7]);
void cmfo_mat7_vec(double out[7], const double M[7][7], const double v[7]);

// =====================================================
//  TENSORES T7
// =====================================================
void cmfo_tensor7(double out[7], const double a[7], const double b[7]);
void cmfo_tensor7_sym(double out[7], const double a[7], const double b[7]);
void cmfo_tensor7_antisym(double out[7], const double a[7], const double b[7]);

// =====================================================
//  PRODUCTOS MATRICES CMFO
// =====================================================
void cmfo_mat7_mul(double out[7][7], const double A[7][7], const double B[7][7]);
void cmfo_mat7_mul_std(double out[7][7], const double A[7][7], const double B[7][7]);

// =====================================================
//  DET Y INVERSA
// =====================================================
double cmfo_mat7_det(const double M[7][7]);
int    cmfo_mat7_inv(double out[7][7], const double M[7][7]);

#endif /* CMFO_CORE_H */

