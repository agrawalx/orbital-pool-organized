// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/**
 * @title MockUSDC
 * @notice Mock USDC token for testing the Orbital Pool
 * @dev Simple ERC20 implementation with 18 decimals
 */
contract MockUSDC is ERC20 {
    uint8 private _decimals;
    
    /**
     * @notice Creates a new MockUSDC token
     * @param name Token name
     * @param symbol Token symbol
     * @param initialSupply Initial supply to mint to deployer
     */
    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply
    ) ERC20(name, symbol) {
        _decimals = 18;
        _mint(msg.sender, initialSupply * 10**_decimals);
    }
    
    /**
     * @notice Returns the number of decimals
     * @return Number of decimals (18)
     */
    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }
    
    /**
     * @notice Mint tokens to a specific address (for testing)
     * @param to Address to mint tokens to
     * @param amount Amount to mint (in tokens, not wei)
     */
    function mint(address to, uint256 amount) external {
        _mint(to, amount * 10**_decimals);
    }
    
    /**
     * @notice Burn tokens from the caller's balance
     * @param amount Amount to burn (in tokens, not wei)
     */
    function burn(uint256 amount) external {
        _burn(msg.sender, amount * 10**_decimals);
    }
}
