//! CMFO Rust SDK
//! Safe Rust bindings to CMFO C ABI

use std::ffi::{CStr, CString};
use std::ptr;

// Re-export raw FFI bindings
pub mod ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// CMFO Result type
pub type Result<T> = std::result::Result<T, Error>;

/// CMFO Error
#[derive(Debug)]
pub enum Error {
    InvalidArg(String),
    OutOfMemory,
    InvalidState,
    License,
    Security,
    Unknown(String),
}

impl From<ffi::cmfo_result_t> for Error {
    fn from(code: ffi::cmfo_result_t) -> Self {
        match code {
            ffi::cmfo_result_t_CMFO_ERROR_INVALID_ARG => Error::InvalidArg("Invalid argument".into()),
            ffi::cmfo_result_t_CMFO_ERROR_OUT_OF_MEMORY => Error::OutOfMemory,
            ffi::cmfo_result_t_CMFO_ERROR_INVALID_STATE => Error::InvalidState,
            ffi::cmfo_result_t_CMFO_ERROR_LICENSE => Error::License,
            ffi::cmfo_result_t_CMFO_ERROR_SECURITY => Error::Security,
            _ => Error::Unknown(format!("Unknown error code: {}", code)),
        }
    }
}

/// 7D Vector
#[derive(Debug, Clone, Copy)]
pub struct Vec7([f64; 7]);

impl Vec7 {
    pub fn new(values: [f64; 7]) -> Self {
        Vec7(values)
    }
    
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

impl From<ffi::cmfo_vec7_t> for Vec7 {
    fn from(v: ffi::cmfo_vec7_t) -> Self {
        Vec7(v.v)
    }
}

impl From<Vec7> for ffi::cmfo_vec7_t {
    fn from(v: Vec7) -> Self {
        ffi::cmfo_vec7_t { v: v.0 }
    }
}

/// CMFO Context
pub struct CMFO {
    ctx: *mut ffi::cmfo_ctx_t,
}

impl CMFO {
    /// Initialize CMFO in study mode
    pub fn new() -> Result<Self> {
        let config = ffi::cmfo_config_t {
            mode: ffi::cmfo_mode_t_CMFO_MODE_STUDY,
            memory_limit_bytes: 0,
            license_key: ptr::null(),
            audit_log_path: ptr::null(),
            flags: 0,
        };
        
        let ctx = unsafe { ffi::cmfo_init(&config as *const _) };
        if ctx.is_null() {
            return Err(Error::Unknown("Failed to initialize CMFO".into()));
        }
        
        Ok(CMFO { ctx })
    }
    
    /// Parse text to 7D vector
    pub fn parse(&self, text: &str) -> Result<Vec7> {
        let c_text = CString::new(text).map_err(|_| Error::InvalidArg("Invalid text".into()))?;
        let mut vec = ffi::cmfo_vec7_t { v: [0.0; 7] };
        
        let result = unsafe {
            ffi::cmfo_parse(self.ctx, c_text.as_ptr(), &mut vec as *mut _)
        };
        
        if result != ffi::cmfo_result_t_CMFO_OK {
            return Err(result.into());
        }
        
        Ok(vec.into())
    }
    
    /// Solve equation
    pub fn solve(&self, equation: &str) -> Result<String> {
        let c_equation = CString::new(equation).map_err(|_| Error::InvalidArg("Invalid equation".into()))?;
        let mut solution_ptr: *mut libc::c_char = ptr::null_mut();
        
        let result = unsafe {
            ffi::cmfo_solve(self.ctx, c_equation.as_ptr(), &mut solution_ptr as *mut _)
        };
        
        if result != ffi::cmfo_result_t_CMFO_OK {
            return Err(result.into());
        }
        
        let solution = unsafe {
            CStr::from_ptr(solution_ptr).to_string_lossy().into_owned()
        };
        
        unsafe { libc::free(solution_ptr as *mut libc::c_void) };
        
        Ok(solution)
    }
    
    /// Compose two vectors
    pub fn compose(&self, v: Vec7, w: Vec7) -> Result<Vec7> {
        let mut result = ffi::cmfo_vec7_t { v: [0.0; 7] };
        let v_ffi: ffi::cmfo_vec7_t = v.into();
        let w_ffi: ffi::cmfo_vec7_t = w.into();
        
        let code = unsafe {
            ffi::cmfo_compose(
                self.ctx,
                &v_ffi as *const _,
                &w_ffi as *const _,
                &mut result as *mut _
            )
        };
        
        if code != ffi::cmfo_result_t_CMFO_OK {
            return Err(code.into());
        }
        
        Ok(result.into())
    }
    
    /// Calculate fractal distance
    pub fn distance(&self, v: Vec7, w: Vec7) -> Result<f64> {
        let mut dist: f64 = 0.0;
        let v_ffi: ffi::cmfo_vec7_t = v.into();
        let w_ffi: ffi::cmfo_vec7_t = w.into();
        
        let code = unsafe {
            ffi::cmfo_distance(
                self.ctx,
                &v_ffi as *const _,
                &w_ffi as *const _,
                &mut dist as *mut _
            )
        };
        
        if code != ffi::cmfo_result_t_CMFO_OK {
            return Err(code.into());
        }
        
        Ok(dist)
    }
}

impl Drop for CMFO {
    fn drop(&mut self) {
        unsafe {
            ffi::cmfo_destroy(self.ctx);
        }
    }
}

unsafe impl Send for CMFO {}
// Note: NOT Sync - each thread needs its own context

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse() {
        let cmfo = CMFO::new().unwrap();
        let vec = cmfo.parse("verdad").unwrap();
        assert_eq!(vec.as_slice().len(), 7);
    }
    
    #[test]
    fn test_solve() {
        let cmfo = CMFO::new().unwrap();
        let solution = cmfo.solve("2x + 3 = 7").unwrap();
        assert!(solution.contains("x = 2"));
    }
}
