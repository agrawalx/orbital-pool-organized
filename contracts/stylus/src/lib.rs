#![cfg_attr(not(any(test, feature = "export-abi")), no_main)]
#![allow(non_snake_case)]
extern crate alloc;

use stylus_sdk::{alloy_primitives::U256, prelude::*, storage::StorageU256};
use alloy_primitives::aliases::U144;
use alloc::vec::Vec;

pub mod orbital_helper;
use orbital_helper::fixed_point::*;
use orbital_helper::parameters::*;
use orbital_helper::invariant;

#[storage]
#[entrypoint]
pub struct OrbitalHelper {
    /// A number stored in contract storage.
    _placeholder: StorageU256,
}

#[public]
impl OrbitalHelper {
    // Forward calls to the modularized functions
    pub fn convert_to_Q96X48(&self, value: U144) -> Result<U144, Vec<u8>> {
        Ok(convert_to_Q96X48(value))
    }

    pub fn convert_from_Q96X48(&self, value: U144) -> Result<U144, Vec<u8>> {
        Ok(convert_from_Q96X48(value))
    }

    pub fn add_Q96X48(&self, a: U144, b: U144) -> Result<U144, Vec<u8>> {
        Ok(add_Q96X48(a, b))
    }

    pub fn sub_Q96X48(&self, a: U144, b: U144) -> Result<U144, Vec<u8>> {
        Ok(sub_Q96X48(a, b))
    }

    pub fn mul_Q96X48(&self, a: U144, b: U144) -> Result<U144, Vec<u8>> {
        Ok(mul_Q96X48(a, b))
    }

    pub fn div_Q96X48(&self, a: U144, b: U144) -> Result<U144, Vec<u8>> {
        Ok(div_Q96X48(a, b))
    }

    pub fn sqrt_Q96X48(&self, y: U144) -> Result<U144, Vec<u8>> {
        Ok(sqrt_Q96X48(y))
    }

    pub fn calculate_radius(&self, reserve: U144) -> Result<U144, Vec<u8>> {
        Ok(calculate_radius(reserve))
    }

    pub fn calculateK(&self, depeg_limit: U144, radius: U144) -> Result<U144, Vec<u8>> {
        Ok(calculateK(depeg_limit, radius))
    }

    pub fn getTickParameters(&self, depeg_limit: U144, reserve: U144) -> Result<(U144, U144), Vec<u8>> {
        Ok(getTickParameters(depeg_limit, reserve))
    }

    pub fn calculateBoundaryTickS(&self, radius: U144, k: U144) -> Result<U144, Vec<u8>> {
        Ok(calculateBoundaryTickS(radius, k))
    }

    pub fn solveQuadraticInvariant(
        &self,
        delta_linear: U144,
        reserves: Vec<U144>,
        token_in_index: U144,
        token_out_index: U144,
        consolidated_radius: U144,
        k_cross: U144,
    ) -> Result<U144, Vec<u8>> {
        Ok(invariant::solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        ))
    }

    pub fn calculate_invariant_simple(
        &self,
        a: i128,
        b: U144,
        r_int: U144,
    ) -> Result<i128, Vec<u8>> {
        Ok(invariant::calculate_invariant_simple(a, b, r_int))
    }

    pub fn signed_div_Q96X48(&self, a: i128, b: i128) -> Result<i128, Vec<u8>> {
        Ok(div_Q96X48_signed(a, b))
    }
    
    pub fn calculate_invariant_derivative(
        &self,
        a: i128,
        b: U144,
        d: U144,
        n: u32,
        x_j: U144,
        sum_reserves: U144,
    ) -> Result<i128, Vec<u8>> {
        Ok(invariant::invariant_derivative(a, b, d, n, x_j, sum_reserves))
    }
    
    pub fn calculate_A_B_D(
        &self,
        sum_reserves: U144,
        sum_reserves_squared: U144,
        n: u32,
        x_j: U144,
        k_bound: U144,
        r_int: U144,
        s_bound: U144,
    ) -> Result<(i128, U144, U144), Vec<u8>> {
        Ok(invariant::calculate_A_B_D(sum_reserves, sum_reserves_squared, n, x_j, k_bound, r_int, s_bound))
    }

    pub fn solve_amount_out(
        &self,
        sum_reserves: U144,
        sum_reserves_squared: U144,
        n: u32,
        k_bound: U144,
        r_int: U144,
        s_bound: U144,
        initial_x_j: U144,
    ) -> Result<U144, Vec<u8>> {
        Ok(invariant::solve_amount_out(sum_reserves, sum_reserves_squared, n, k_bound, r_int, s_bound, initial_x_j))
    }
}