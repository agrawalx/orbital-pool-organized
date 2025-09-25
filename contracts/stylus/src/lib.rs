#![cfg_attr(not(feature = "export-abi"), no_main)]
#![allow(non_snake_case)] // Allows camelCase function names to match Solidity.

#[macro_use]
extern crate alloc;

mod orbital_helper;

use alloc::vec::Vec;
use stylus_sdk::{alloy_primitives::U256, prelude::*, storage::StorageU256};
use alloy_primitives::aliases::U144;

use orbital_helper::fixed_point::*;
use orbital_helper::parameters::*;
use orbital_helper::invariant;

// Define some persistent storage using the Solidity ABI.
// `Counter` will be the entrypoint.
#[storage]
#[entrypoint]
pub struct OrbitalHelper {
    /// A number stored in contract storage.
    _placeholder: StorageU256,
}

#[public]
impl OrbitalHelper {
    // Forward calls to the modularized functions
    pub fn convert_to_Q96X48(value: U144) -> U144 {
        convert_to_Q96X48(value)
    }

    pub fn convert_from_Q96X48(value: U144) -> U144 {
        convert_from_Q96X48(value)
    }

    pub fn add_Q96X48(a: U144, b: U144) -> U144 {
        add_Q96X48(a, b)
    }

    pub fn sub_Q96X48(a: U144, b: U144) -> U144 {
        sub_Q96X48(a, b)
    }

    pub fn mul_Q96X48(a: U144, b: U144) -> U144 {
        mul_Q96X48(a, b)
    }

    pub fn div_Q96X48(a: U144, b: U144) -> U144 {
        div_Q96X48(a, b)
    }

    pub fn sqrt_Q96X48(y: U144) -> U144 {
        sqrt_Q96X48(y)
    }

    pub fn calculate_radius(reserve: U144) -> U144 {
        calculate_radius(reserve)
    }

    pub fn calculateK(depeg_limit: U144, radius: U144) -> U144 {
        calculateK(depeg_limit, radius)
    }

    pub fn getTickParameters(depeg_limit: U144, reserve: U144) -> (U144, U144) {
        getTickParameters(depeg_limit, reserve)
    }

    pub fn calculateBoundaryTickS(radius: U144, k: U144) -> U144 {
        calculateBoundaryTickS(radius, k)
    }

    pub fn solveQuadraticInvariant(
        delta_linear: U144,
        reserves: Vec<U144>,
        token_in_index: U144,
        token_out_index: U144,
        consolidated_radius: U144,
        k_cross: U144,
    ) -> U144 {
        invariant::solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        )
    }

    pub fn solve_amount_out(
        &self,
        reserves: Vec<U144>,
        amount_in: U144,
        token_in_index: U256,
        token_out_index: U256,
        k_bound: U144,
        r_int: U144,
        s_bound: U144,
    ) -> U144 {
        invariant::solve_amount_out(
            reserves,
            amount_in,
            token_in_index,
            token_out_index,
            k_bound,
            r_int,
            s_bound,
        )
    }
}
