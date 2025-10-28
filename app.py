# =============================================================================
# HEDGING MODULE - STREAMLIT APPLICATION
# =============================================================================
# This application manages hedging positions for commodity trading
# It supports two teams: Team A (Document Tagging) and Team B (Hedging Management)
import warnings
warnings.filterwarnings("ignore")
# Import required libraries
import streamlit as st          # Main Streamlit library for web app interface
import altair as alt
import uuid                     # Generate unique identifiers for positions and entries
from datetime import date, timedelta, datetime  # Handle dates and time calculations
import pandas as pd             # Data manipulation and display
import random                   # Generate random numbers for position IDs
from typing import Dict, List, Optional, Tuple  # Type hints for better code clarity
from constants_variable import EXCHANGES, EXCHANGE_CURRENCIES  # Import exchange constants
import constants_variable
import utils                    # Utility functions for database operations
import time
import datetime
import secrets



# =============================================================================
# DATA MODELS - CORE CLASSES FOR HEDGING POSITIONS
# =============================================================================

class HedgePosition:
    """
    Represents a single hedging position that can contain multiple buy/sell entries
    A position starts with either a Paper Buy or Paper Sell and can be extended with rollovers
    """
    
    def __init__(self, position_id: str, bill_id: str, commodity: str):
        """
        Initialize a new hedging position
        
        Args:
            position_id: Unique identifier for this position
            bill_id: Reference to the original bill (if applicable)
            commodity: Type of commodity (e.g., Copper, Aluminium, Zinc, Nickel)
        """
        self.position_id = position_id      # Unique position identifier
        self.bill_id = bill_id              # Reference to original bill
        self.commodity = commodity          # Commodity type
        self.entries: List[Dict] = []       # List of all buy/sell entries in this position
        self.status = "OPEN"                # Position status: OPEN, CLOSED, or ROLLED
        
    def add_entry(self, entry_type: str, **kwargs):
        """
        Add a new buy or sell entry to this position
        
        Args:
            entry_type: Either 'buy' or 'sell'
            **kwargs: Additional entry details (quantity, rate, exchange, etc.)
        """
        # Create a new entry with unique ID and current timestamp
        entry = {
            'entry_id': kwargs.get('entry_id', str(uuid.uuid4())),  # Use provided entry_id or generate new one
            'type': entry_type,             # Entry type (buy/sell)
            'timestamp': date.today(),      # Current date
            **kwargs                        # Include all other provided details
        }
        self.entries.append(entry)          # Add to position's entry list
        
    def get_latest_entry(self) -> Optional[Dict]:
        """
        Get the most recent entry in this position
        
        Returns:
            Latest entry dictionary or None if no entries exist
        """
        return self.entries[-1] if self.entries else None
        
    def get_open_quantity(self) -> float:
        """
        Calculate the current open quantity for this position
        
        Returns:
            Positive number = net buy position (need to sell to close)
            Negative number = net sell position (need to buy to close)
            Zero = position is closed
        """
        # Sum all quantities (positive for buy, negative for sell)
        total_qty = sum(e.get('quantity', 0) for e in self.entries)
        return total_qty
        
    def is_closed(self) -> bool:
        """
        Check if this position is fully closed (no open quantity)
        
        Returns:
            True if position is closed, False if still open
        """
        # Use small threshold (0.001) to handle floating point precision issues
        return abs(self.get_open_quantity()) < 0.001

class HedgingModule:
    """
    Main application class that manages all hedging operations
    Handles positions, bills, invoices, and their relationships
    """
    
    def __init__(self):
        """
        Initialize the hedging module with empty data structures
        """
        self.positions: Dict[str, HedgePosition] = {}    # All hedging positions by ID
        self.bills: List[Dict] = []                      # List of supplier bills from OMS
        self.invoices: List[Dict] = []                   # List of customer invoices from OMS
        self.bill_tags: Dict[str, str] = {}              # Maps bill_id to paper_sell_id
        self.invoice_tags: Dict[str, str] = {}           # Maps invoice_id to paper_buy_id
        
    def create_position(self, bill_id: str, commodity: str) -> str:
        """
        Create a new hedging position
        
        Args:
            bill_id: Reference to the bill (or generated ID for new positions)
            commodity: Type of commodity
            
        Returns:
            Unique position ID
        """
        # Generate a unique position ID with HEDGE prefix
        position_id = f"HEDGE_{random.randint(100000, 999999)}"
        # Create new position and store it
        self.positions[position_id] = HedgePosition(position_id, bill_id, commodity)
        return position_id
        
    def add_paper_sell(self, position_id: str, quantity: float, exchange: str, 
                       rate: float, due_date: date, **kwargs) -> str:
        """
        Add a paper sell entry to an existing position
        
        Args:
            position_id: ID of the position to add entry to
            quantity: Quantity to sell in MT
            exchange: Trading exchange (MCX/LME)
            rate: Selling rate/price
            due_date: Expiry date for the sell position
            **kwargs: Additional entry details
            
        Returns:
            Entry ID if successful, None if position not found
        """
        # Get the position to add entry to
        position = self.positions.get(position_id)
        if not position:
            return None
            
        # Generate unique entry ID
        entry_id = str(uuid.uuid4())
        # Add sell entry to position
        # For sell entries, store quantity as negative to properly represent sell position
        position.add_entry('sell', 
                         entry_id=entry_id,
                         quantity=-quantity,  # Negative quantity for sell entries
                         exchange=exchange,
                         rate=rate,
                         due_date=due_date,
                         **kwargs)
        return entry_id
        
    def add_paper_buy(self, position_id: str, quantity: float, exchange: str,
                      rate: float, **kwargs) -> str:
        """
        Add a paper buy entry to an existing position
        
        Args:
            position_id: ID of the position to add entry to
            quantity: Quantity to buy in MT
            exchange: Trading exchange (MCX/LME)
            rate: Buying rate/price
            **kwargs: Additional entry details
            
        Returns:
            Entry ID if successful, None if position not found
        """
        # Get the position to add entry to
        position = self.positions.get(position_id)
        if not position:
            return None
            
        # Generate unique entry ID
        entry_id = str(uuid.uuid4())
        # Add buy entry to position
        position.add_entry('buy',
                         entry_id=entry_id,
                         quantity=quantity,
                         exchange=exchange,
                         rate=rate,
                         **kwargs)
        return entry_id
        
    def get_open_positions(self) -> List[HedgePosition]:
        """
        Get all positions that are not fully closed and have at least one entry
        
        Returns:
            List of open hedging positions
        """
        return [pos for pos in self.positions.values() if not pos.is_closed() and len(pos.entries) > 0]
        
    def get_positions_by_expiry(self, days_threshold: int = 5) -> List[Dict]:
        """
        Get all entries (both buy and sell) that are close to expiry
        
        Args:
            days_threshold: Number of days to look ahead (default: 5 days)
            
        Returns:
            List of entries close to expiry with position details
        """
        # Calculate the threshold date
        threshold_date = date.today() + timedelta(days=days_threshold)
        expiring_entries = []
        
        # Check all positions and their entries
        for position in self.positions.values():
            for entry in position.entries:
                # If entry has a due date and is within threshold
                if entry.get('due_date') and entry['due_date'] <= threshold_date:
                    # Create summary of expiring entry
                    expiring_entries.append({
                        'position_id': position.position_id,
                        'commodity': position.commodity,
                        'entry_type': entry['type'],           # 'buy' or 'sell'
                        'due_date': entry['due_date'],
                        'quantity': entry['quantity'],
                        'exchange': entry['exchange'],
                        'rate': entry['rate'],
                        'supplier_name': entry.get('supplier_name'),  # For sell entries
                        'customer_name': entry.get('customer_name'),  # For buy entries
                        'num_lots': entry.get('num_lots'),           # MCX specific
                        'lot_size': entry.get('lot_size')            # MCX specific
                    })
                        
        return expiring_entries
    
    def settle_positions(self, buy_position_id: str, sell_position_id: str, quantity: float) -> bool:
        """Settle positions by creating offsetting entries that close the positions"""
        try:
            # Get the positions
            buy_position = self.positions.get(buy_position_id)
            sell_position = self.positions.get(sell_position_id)
            
            if not buy_position or not sell_position:
                return False
            
            # Validate quantities
            buy_open_qty = buy_position.get_open_quantity()
            sell_open_qty = abs(sell_position.get_open_quantity())
            
            if quantity > buy_open_qty or quantity > sell_open_qty:
                return False
            
            # Create settlement entries that PROPERLY OFFSET the open positions
            # Settlement entry for BUY position (adds a sell to close the buy)
            settlement_entry_for_buy = {
                'entry_id': f"SB_{random.randint(100000, 999999)}",
                'type': 'sell',  # SELL to close BUY position
                'quantity': -quantity,  # Negative quantity to offset the buy
                'exchange': 'MCX',
                'rate': 800.0,
                'timestamp': date.today(),
                'currency': 'INR',
                'conversion_rate': 1.0,
                'supplier_name': 'SETTLEMENT_ENTRY',
                'num_lots': 1,
                'lot_size': quantity,
                'transaction_date': date.today(),
                'due_date': date.today() + timedelta(days=30)
            }
            
            # Settlement entry for SELL position (adds a buy to close the sell)
            settlement_entry_for_sell = {
                'entry_id': f"SS_{random.randint(100000, 999999)}",
                'type': 'buy',  # BUY to close SELL position
                'quantity': quantity,  # Positive quantity to offset the sell
                'exchange': 'MCX',
                'rate': 800.0,
                'timestamp': date.today(),
                'currency': 'INR',
                'conversion_rate': 1.0,
                'customer_name': 'SETTLEMENT_ENTRY',
                'num_lots': 1,
                'lot_size': quantity,
                'transaction_date': date.today(),
                'due_date': date.today() + timedelta(days=30)
            }
            
            # Add settlement entries to positions
            buy_position.entries.append(settlement_entry_for_buy)
            sell_position.entries.append(settlement_entry_for_sell)
            
            return True
        except Exception as e:
            return False


# --- Mock Data Generation ---
def generate_mock_data():
    """Generate sample bills, invoices, and initial positions"""
    bills = [
        # Copper Bills
        {'id': 'BILL001', 'commodity': 'Copper', 'quantity': 100, 'supplier': 'Copper Supplier A', 'date': date.today() - timedelta(days=10)},
        {'id': 'BILL004', 'commodity': 'Copper', 'quantity': 75, 'supplier': 'Copper Supplier D', 'date': date.today() - timedelta(days=3)},
        {'id': 'BILL007', 'commodity': 'Copper', 'quantity': 125, 'supplier': 'Copper Supplier G', 'date': date.today() - timedelta(days=1)},
        
        # Aluminium Bills
        {'id': 'BILL002', 'commodity': 'Aluminium', 'quantity': 80, 'supplier': 'Aluminium Supplier B', 'date': date.today() - timedelta(days=8)},
        {'id': 'BILL005', 'commodity': 'Aluminium', 'quantity': 120, 'supplier': 'Aluminium Supplier E', 'date': date.today() - timedelta(days=2)},
        {'id': 'BILL008', 'commodity': 'Aluminium', 'quantity': 95, 'supplier': 'Aluminium Supplier H', 'date': date.today() - timedelta(days=4)},
        
        # Zinc Bills
        {'id': 'BILL003', 'commodity': 'Zinc', 'quantity': 60, 'supplier': 'Zinc Supplier C', 'date': date.today() - timedelta(days=5)},
        {'id': 'BILL006', 'commodity': 'Zinc', 'quantity': 90, 'supplier': 'Zinc Supplier F', 'date': date.today() - timedelta(days=1)},
        {'id': 'BILL009', 'commodity': 'Zinc', 'quantity': 110, 'supplier': 'Zinc Supplier I', 'date': date.today() - timedelta(days=6)},
    ]
    
    invoices = [
        # Copper Invoices
        {'id': 'INV001', 'customer': 'Copper Customer X', 'commodity': 'Copper', 'quantity': 50, 'date': date.today() - timedelta(days=2)},
        {'id': 'INV004', 'customer': 'Copper Customer W', 'commodity': 'Copper', 'quantity': 65, 'date': date.today() - timedelta(days=3)},
        {'id': 'INV007', 'customer': 'Copper Customer P', 'commodity': 'Copper', 'quantity': 85, 'date': date.today() - timedelta(days=5)},
        
        # Aluminium Invoices
        {'id': 'INV002', 'customer': 'Aluminium Customer Y', 'commodity': 'Aluminium', 'quantity': 40, 'date': date.today() - timedelta(days=1)},
        {'id': 'INV005', 'customer': 'Aluminium Customer V', 'commodity': 'Aluminium', 'quantity': 55, 'date': date.today() - timedelta(days=4)},
        {'id': 'INV008', 'customer': 'Aluminium Customer Q', 'commodity': 'Aluminium', 'quantity': 70, 'date': date.today() - timedelta(days=7)},
        
        # Zinc Invoices
        {'id': 'INV003', 'customer': 'Zinc Customer Z', 'commodity': 'Zinc', 'quantity': 30, 'date': date.today()},
        {'id': 'INV006', 'customer': 'Zinc Customer U', 'commodity': 'Zinc', 'quantity': 45, 'date': date.today() - timedelta(days=5)},
        {'id': 'INV009', 'customer': 'Zinc Customer R', 'commodity': 'Zinc', 'quantity': 60, 'date': date.today() - timedelta(days=3)},
    ]
    
    return bills, invoices

def generate_dummy_paper_entries(hedging_module: HedgingModule):
    """Generate dummy paper entries for testing settlement functionality"""
    print("🔧 Creating dummy paper entries for testing...")
    
    # =============================================================================
    # COPPER ENTRIES (100 MT Buy, 25 MT Sell, 50 MT Sell)
    # =============================================================================
    # Create Paper Buy Entry: 100 MT Copper
    buy_position_id_copper = hedging_module.create_position("DUMMY_BILL_COPPER_001", "Copper")
    buy_entry_id_copper = hedging_module.add_paper_buy(
        position_id=buy_position_id_copper,
        quantity=100.0,  # 100 MT
        exchange="MCX",
        rate=850.0,
        currency="INR",
        conversion_rate=1.0,
        customer_name="COPPER CUSTOMER A",
        num_lots=4,
        lot_size=25.0,
        transaction_date=date.today(),
        due_date=date.today() + timedelta(days=30)
    )
    print(f"✅ Created Copper Paper Buy: {buy_position_id_copper} - 100 MT")
    
    # Create Paper Sell Entry 1: 25 MT Copper
    sell_position_id_copper_1 = hedging_module.create_position("DUMMY_BILL_COPPER_002", "Copper")
    sell_entry_id_copper_1 = hedging_module.add_paper_sell(
        position_id=sell_position_id_copper_1,
        quantity=25.0,  # 25 MT
        exchange="MCX",
        rate=860.0,
        due_date=date.today() + timedelta(days=25),
        currency="INR",
        conversion_rate=1.0,
        supplier_name="COPPER SUPPLIER A",
        num_lots=1,
        lot_size=25.0,
        transaction_date=date.today()
    )
    print(f"✅ Created Copper Paper Sell 1: {sell_position_id_copper_1} - 25 MT")
    
    # Create Paper Sell Entry 2: 50 MT Copper
    sell_position_id_copper_2 = hedging_module.create_position("DUMMY_BILL_COPPER_003", "Copper")
    sell_entry_id_copper_2 = hedging_module.add_paper_sell(
        position_id=sell_position_id_copper_2,
        quantity=50.0,  # 50 MT
        exchange="MCX",
        rate=865.0,
        due_date=date.today() + timedelta(days=20),
        currency="INR",
        conversion_rate=1.0,
        supplier_name="COPPER SUPPLIER B",
        num_lots=2,
        lot_size=25.0,
        transaction_date=date.today()
    )
    print(f"✅ Created Copper Paper Sell 2: {sell_position_id_copper_2} - 50 MT")
    
    # =============================================================================
    # ALUMINIUM ENTRIES (100 MT Buy, 25 MT Sell, 50 MT Sell)
    # =============================================================================
    # Create Paper Buy Entry: 100 MT Aluminium
    buy_position_id_aluminium = hedging_module.create_position("DUMMY_BILL_ALUMINIUM_001", "Aluminium")
    buy_entry_id_aluminium = hedging_module.add_paper_buy(
        position_id=buy_position_id_aluminium,
        quantity=100.0,  # 100 MT
        exchange="MCX",
        rate=220.0,
        currency="INR",
        conversion_rate=1.0,
        customer_name="ALUMINIUM CUSTOMER A",
        num_lots=5,
        lot_size=20.0,
        transaction_date=date.today(),
        due_date=date.today() + timedelta(days=30)
    )
    print(f"✅ Created Aluminium Paper Buy: {buy_position_id_aluminium} - 100 MT")
    
    # Create Paper Sell Entry 1: 25 MT Aluminium
    sell_position_id_aluminium_1 = hedging_module.create_position("DUMMY_BILL_ALUMINIUM_002", "Aluminium")
    sell_entry_id_aluminium_1 = hedging_module.add_paper_sell(
        position_id=sell_position_id_aluminium_1,
        quantity=25.0,  # 25 MT
        exchange="MCX",
        rate=225.0,
        due_date=date.today() + timedelta(days=25),
        currency="INR",
        conversion_rate=1.0,
        supplier_name="ALUMINIUM SUPPLIER A",
        num_lots=1,
        lot_size=25.0,
        transaction_date=date.today()
    )
    print(f"✅ Created Aluminium Paper Sell 1: {sell_position_id_aluminium_1} - 25 MT")
    
    # Create Paper Sell Entry 2: 50 MT Aluminium
    sell_position_id_aluminium_2 = hedging_module.create_position("DUMMY_BILL_ALUMINIUM_003", "Aluminium")
    sell_entry_id_aluminium_2 = hedging_module.add_paper_sell(
        position_id=sell_position_id_aluminium_2,
        quantity=50.0,  # 50 MT
        exchange="MCX",
        rate=230.0,
        due_date=date.today() + timedelta(days=20),
        currency="INR",
        conversion_rate=1.0,
        supplier_name="ALUMINIUM SUPPLIER B",
        num_lots=2,
        lot_size=25.0,
        transaction_date=date.today()
    )
    print(f"✅ Created Aluminium Paper Sell 2: {sell_position_id_aluminium_2} - 50 MT")
    
    # =============================================================================
    # ZINC ENTRIES (100 MT Buy, 25 MT Sell, 50 MT Sell)
    # =============================================================================
    # Create Paper Buy Entry: 100 MT Zinc
    buy_position_id_zinc = hedging_module.create_position("DUMMY_BILL_ZINC_001", "Zinc")
    buy_entry_id_zinc = hedging_module.add_paper_buy(
        position_id=buy_position_id_zinc,
        quantity=100.0,  # 100 MT
        exchange="MCX",
        rate=320.0,
        currency="INR",
        conversion_rate=1.0,
        customer_name="ZINC CUSTOMER A",
        num_lots=5,
        lot_size=20.0,
        transaction_date=date.today(),
        due_date=date.today() + timedelta(days=30)
    )
    print(f"✅ Created Zinc Paper Buy: {buy_position_id_zinc} - 100 MT")
    
    # Create Paper Sell Entry 1: 25 MT Zinc
    sell_position_id_zinc_1 = hedging_module.create_position("DUMMY_BILL_ZINC_002", "Zinc")
    sell_entry_id_zinc_1 = hedging_module.add_paper_sell(
        position_id=sell_position_id_zinc_1,
        quantity=25.0,  # 25 MT
        exchange="MCX",
        rate=325.0,
        due_date=date.today() + timedelta(days=25),
        currency="INR",
        conversion_rate=1.0,
        supplier_name="ZINC SUPPLIER A",
        num_lots=1,
        lot_size=25.0,
        transaction_date=date.today()
    )
    print(f"✅ Created Zinc Paper Sell 1: {sell_position_id_zinc_1} - 25 MT")
    
    # Create Paper Sell Entry 2: 50 MT Zinc
    sell_position_id_zinc_2 = hedging_module.create_position("DUMMY_BILL_ZINC_003", "Zinc")
    sell_entry_id_zinc_2 = hedging_module.add_paper_sell(
        position_id=sell_position_id_zinc_2,
        quantity=50.0,  # 50 MT
        exchange="MCX",
        rate=330.0,
        due_date=date.today() + timedelta(days=20),
        currency="INR",
        conversion_rate=1.0,
        supplier_name="ZINC SUPPLIER B",
        num_lots=2,
        lot_size=25.0,
        transaction_date=date.today()
    )
    print(f"✅ Created Zinc Paper Sell 2: {sell_position_id_zinc_2} - 50 MT")
    
    print(f"🎯 Total positions created: {len(hedging_module.positions)}")
    print(f"📊 Ready for testing settlement functionality!")
    print(f"📋 Created for each commodity: 1x 100MT Paper Buy, 1x 25MT Paper Sell, 1x 50MT Paper Sell")
    
    return {
        'copper_buy_position': buy_position_id_copper,
        'copper_sell_position_1': sell_position_id_copper_1,
        'copper_sell_position_2': sell_position_id_copper_2,
        'aluminium_buy_position': buy_position_id_aluminium,
        'aluminium_sell_position_1': sell_position_id_aluminium_1,
        'aluminium_sell_position_2': sell_position_id_aluminium_2,
        'zinc_buy_position': buy_position_id_zinc,
        'zinc_sell_position_1': sell_position_id_zinc_1,
        'zinc_sell_position_2': sell_position_id_zinc_2
    }

def fifo_match_invoice(buy_df, sell_df,
               buy_qty_col='invoice_open_quantity',
               sell_qty_col='open_qty',
               buy_id_col='position_id',
               sell_id_col='invoicenumber',
               product_col='product'):
    """
    Perform FIFO matching of sell rows against buy rows.
    Returns (matches_df, buy_df_with_remaining, sell_df_with_remaining)
    """
    # Work on copies and reset indices to ensure positional .iloc/.at works
    buys = buy_df.copy().reset_index(drop=True)
    sells = sell_df.copy().reset_index(drop=True)

    # Ensure numeric quantities
    buys[buy_qty_col] = pd.to_numeric(buys[buy_qty_col], errors='coerce').fillna(0)
    sells[sell_qty_col] = pd.to_numeric(sells[sell_qty_col], errors='coerce').fillna(0)

    # Initialize remaining columns
    buys['remaining_qty'] = buys[buy_qty_col].astype(float)
    sells['remaining_qty'] = sells[sell_qty_col].astype(float)

    matches = []
    buy_index = 0
    n_buys = len(buys)

    for sell_idx, sell_row in sells.iterrows():
        sell_qty_left = sells.at[sell_idx, 'remaining_qty']

        # Skip zero sells quickly
        if sell_qty_left <= 0:
            continue

        # Walk buys FIFO
        while sell_qty_left > 0 and buy_index < n_buys:
            buy_qty_left = buys.at[buy_index, 'remaining_qty']

            # If this buy is exhausted, advance
            if buy_qty_left <= 0:
                buy_index += 1
                continue

            matched_qty = min(sell_qty_left, buy_qty_left)

            matches.append({
                buy_id_col: buys.at[buy_index, buy_id_col],
                sell_id_col: sells.at[sell_idx, sell_id_col],
                'product': buys.at[buy_index, product_col] if product_col in buys.columns else None,
                'tagged_qty': matched_qty,
                'paper_remaining_after': buy_qty_left - matched_qty,
                'invoice_remaining_after': sell_qty_left - matched_qty
            })

            # Update remaining quantities on both sides
            buys.at[buy_index, 'remaining_qty'] = buy_qty_left - matched_qty
            sell_qty_left -= matched_qty
            sells.at[sell_idx, 'remaining_qty'] = sell_qty_left

            # If buy exhausted, move to next buy
            if buys.at[buy_index, 'remaining_qty'] <= 0:
                buy_index += 1

        # if we exit while and sell_qty_left > 0, then sells remains partially/unmatched

    matches_df = pd.DataFrame(matches)

    if not matches_df.empty:
        if 'tagged_qty' in matches_df.columns:
            matches_df['tagged_qty'] = matches_df['tagged_qty'].astype(float)

    buys['remaining_qty'] = buys['remaining_qty'].astype(float)
    sells['remaining_qty'] = sells['remaining_qty'].astype(float)

    return matches_df, buys, sells


def fifo_match_bill(buy_df, sell_df,
               buy_qty_col='bill_open_quantity',
               sell_qty_col='open_qty',
               buy_id_col='position_id',
               sell_id_col='billnumber',
               product_col='product'):
    """
    Perform FIFO matching of sell rows against buy rows.
    Returns (matches_df, buy_df_with_remaining, sell_df_with_remaining)
    """
    # Work on copies and reset indices to ensure positional .iloc/.at works
    buys = buy_df.copy().reset_index(drop=True)
    sells = sell_df.copy().reset_index(drop=True)

    # Ensure numeric quantities
    buys[buy_qty_col] = pd.to_numeric(buys[buy_qty_col], errors='coerce').fillna(0)
    sells[sell_qty_col] = pd.to_numeric(sells[sell_qty_col], errors='coerce').fillna(0)

    # Initialize remaining columns
    buys['remaining_qty'] = buys[buy_qty_col].astype(float)
    sells['remaining_qty'] = sells[sell_qty_col].astype(float)

    matches = []
    buy_index = 0
    n_buys = len(buys)

    for sell_idx, sell_row in sells.iterrows():
        sell_qty_left = sells.at[sell_idx, 'remaining_qty']

        # Skip zero sells quickly
        if sell_qty_left <= 0:
            continue

        # Walk buys FIFO
        while sell_qty_left > 0 and buy_index < n_buys:
            buy_qty_left = buys.at[buy_index, 'remaining_qty']

            # If this buy is exhausted, advance
            if buy_qty_left <= 0:
                buy_index += 1
                continue

            matched_qty = min(sell_qty_left, buy_qty_left)

            matches.append({
                buy_id_col: buys.at[buy_index, buy_id_col],
                sell_id_col: sells.at[sell_idx, sell_id_col],
                'product': buys.at[buy_index, product_col] if product_col in buys.columns else None,
                'tagged_qty': matched_qty,
                'paper_remaining_after': buy_qty_left - matched_qty,
                'bill_remaining_after': sell_qty_left - matched_qty
            })

            # Update remaining quantities on both sides
            buys.at[buy_index, 'remaining_qty'] = buy_qty_left - matched_qty
            sell_qty_left -= matched_qty
            sells.at[sell_idx, 'remaining_qty'] = sell_qty_left

            # If buy exhausted, move to next buy
            if buys.at[buy_index, 'remaining_qty'] <= 0:
                buy_index += 1

        # if we exit while and sell_qty_left > 0, then sells remains partially/unmatched

    matches_df = pd.DataFrame(matches)

    if not matches_df.empty:
        if 'tagged_qty' in matches_df.columns:
            matches_df['tagged_qty'] = matches_df['tagged_qty'].astype(float)

    buys['remaining_qty'] = buys['remaining_qty'].astype(float)
    sells['remaining_qty'] = sells['remaining_qty'].astype(float)

    return matches_df, buys, sells

def generate_entity_id(existing_ids=None):
    """
    Generate a unique 24-character hex ID not in existing_ids (if provided).
    """
    existing_ids = set(existing_ids or [])
    while True:
        new_id = secrets.token_hex(12)  # 12 bytes = 24 hex chars
        if new_id not in existing_ids:
            return new_id

# --- UI Components ---
def render_invoice_tagging_tab(q1,q2,q3):
    """Render the invoice tagging tab"""
    st.header("📄 Invoice Tagging")
    invoice_df = utils.get_question_data(q1)
    paper_buy_df = utils.get_table_data(q2)
    invoice_tagged_zata_df = utils.get_table_data(q3)
    
    paper_buy_df = paper_buy_df[(paper_buy_df['invoice_open_quantity'] > 0) & (paper_buy_df['artefact_status'] == 'OPEN')]
    
    if invoice_df.empty:
        st.warning("No invoices available. Please upload invoices from OMS system.")
        return
    
    # =============================================================================
    # COMMODITY FILTER FOR INVOICES
    # =============================================================================
    # Get all unique commodities from invoices
    all_invoice_commodities = set(list(invoice_df['product']))
    
    if all_invoice_commodities:
        # Add "All Commodities" option
        commodity_options = ["All Commodities"] + sorted(list(all_invoice_commodities))
        
        # Create commodity filter
        col1, col2 = st.columns([1,3])
        with col1:
            selected_commodity = st.selectbox(
                "Filter Invoices by Commodity:",
                options=commodity_options,
                index=0,
                key="invoice_commodity_filter"
            )
        with col2:
            # Show filter status
            if selected_commodity != "All Commodities":
                st.info(f"📊 Showing invoices for: **{selected_commodity}**")
            else:
                commodities_str = ", ".join(sorted(all_invoice_commodities))
                st.info(f"📊 Showing invoices for: **All Commodities** ({commodities_str})")
    else:
        selected_commodity = "All Commodities"
        st.info("No invoices available to filter")
    
    st.markdown("---")
        
    # Invoice selection (filtered by commodity)
    if selected_commodity != "All Commodities":
        # Filter invoices by selected commodity
        invoice_df = invoice_df[invoice_df['product'] == selected_commodity]
        paper_buy_df = paper_buy_df[paper_buy_df['product'] == selected_commodity]
        if invoice_df.empty:
            st.warning(f"No {selected_commodity} invoices available.")
            return

    # Create filtered invoice options

    available_buys = []
    available_buys = paper_buy_df[paper_buy_df['invoice_open_quantity'] > 0]
    invoice_df['filter'] = invoice_df['invoicenumber'].astype(str) + ' with open qty ' + invoice_df['open_qty'].astype(str) + ' ' + invoice_df['item_unit'] +' '+ invoice_df['product']
    available_buys['filter'] = available_buys['position_id'] + ' with open qty ' + available_buys['invoice_open_quantity'].astype(str) +' MT '+ available_buys['product']
    col1,col2 = st.columns([1,1])
    invoice_df_info = invoice_df.copy()
    available_buys_info = available_buys.copy()
    invoice_df_info = invoice_df_info[invoice_df_info['open_qty'] != 0]
    available_buys_info = available_buys_info[available_buys_info['invoice_open_quantity'] != 0]
    invoice_df_info = invoice_df_info[invoice_df_info['customer_name'].isin(available_buys_info['customer_group_name'].unique())]
    available_buys_info = available_buys_info[available_buys_info['customer_group_name'].isin(invoice_df_info['customer_name'].unique())]
    with col1:
        st.subheader("Available Invoices")
        if not invoice_df.empty:
            col11, col22, col33 = st.columns([1, 1, 1])

            # --- Build filters from full dataset ---
            with col11:
                invoice_number_filter = st.multiselect(
                    "Filter Invoices by Invoice Number",
                    sorted(invoice_df_info['invoicenumber'].unique()),
                    key="invoice_number_filter"
                )

            with col22:
                invoice_customer_filter = st.multiselect(
                    "Filter Invoices by Customer Name",
                    sorted(invoice_df_info['customer_name'].unique()),
                    key="invoice_customer_filter"
                )

            with col33:
                # Convert and clean up date column
                invoice_df_info['invoicedate'] = pd.to_datetime(invoice_df_info['invoicedate'], errors='coerce').dt.date

                # Compute valid date range
                min_date = invoice_df_info['invoicedate'].min()
                max_date = invoice_df_info['invoicedate'].max()

                # Handle case where dates are missing or invalid
                if pd.isna(min_date) or pd.isna(max_date):
                    min_date = datetime.date.today()
                    max_date = datetime.date.today()

                # Streamlit date input
                start_date, end_date = st.date_input(
                    "Filter Invoices by Date Range",
                    [min_date, max_date],
                    key="invoice_date_filter"
                )


            if invoice_number_filter:
                invoice_df_info = invoice_df_info[invoice_df_info['invoicenumber'].isin(invoice_number_filter)]

            if invoice_customer_filter:
                invoice_df_info = invoice_df_info[invoice_df_info['customer_name'].isin(invoice_customer_filter)]

            if start_date and end_date:
                invoice_df_info = invoice_df_info[
                    (invoice_df_info['invoicedate'] >= start_date) &
                    (invoice_df_info['invoicedate'] <= end_date)
                ]

    with col2:
        if not available_buys.empty:
            st.subheader("Available Paper Buy positions")
            col11,col22,col33 = st.columns([1,1,1])
            with col11:
                paper_buy_number_filter = st.multiselect(
                    "Filter paper buy by position_id",
                    sorted(available_buys_info['position_id'].unique()),
                    key="position_id_filter"
                )

            with col22:
                paper_buy_customer_filter = st.multiselect(
                    "Filter paper buy by Customer Name",
                    sorted(available_buys_info['customer_name'].unique()),
                    key="paper_buy_customer_filter"
                )

            with col33:
                available_buys_info['transaction_date'] = pd.to_datetime(
                    available_buys_info['transaction_date'], errors='coerce'
                )

                if available_buys_info['transaction_date'].notna().any():
                    min_date = available_buys_info['transaction_date'].min().date()
                    max_date = available_buys_info['transaction_date'].max().date()
                    start_date, end_date = st.date_input(
                        "Filter Paper Buy by Date Range",
                        [min_date, max_date],
                        key="paper_buy_date_filter"
                    )
                else:
                    start_date, end_date = None, None

            if paper_buy_number_filter:
                available_buys_info = available_buys_info[available_buys_info['transaction_date'].isin(invoice_number_filter)]

            if paper_buy_customer_filter:
                available_buys_info = available_buys_info[available_buys_info['transaction_date'].isin(invoice_customer_filter)]

            if start_date and end_date:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                available_buys_info = available_buys_info[
                    (available_buys_info['transaction_date'] >= start_date) &
                    (available_buys_info['transaction_date'] <= end_date)
                ]

    with col1:
        st.dataframe(invoice_df_info[['invoicenumber','invoicedate','customer_name', 'product','item_name','open_qty','item_unit','open_qty_value']], hide_index=True) 
    
    with col2:
        st.dataframe(available_buys_info[['position_id','product', 'artefact_status','customer_name', 'invoice_open_quantity','price_rate','expiry_date']],hide_index=True)
    
    col1,col2 = st.columns([1,1])
    with col1:
        selected_invoice = st.multiselect("Select Invoice to Tag (Multiple selection allowed)", list(invoice_df_info['filter'].unique()), key="invoice_select")
        formatted_text = "\n\n".join(str(item) for item in selected_invoice) 
        st.markdown(f"{formatted_text}")
    
    with col2:
        selected_buys = st.multiselect("Select Paper Buy Entries to Tag (Multiple selection allowed)", list(available_buys_info['filter'].unique()))
        formatted_text_buy = "\n\n".join(str(item) for item in selected_buys) 
        st.markdown(f"{formatted_text_buy}")
    
    with col1:
        hedge_qty = st.number_input("Enter Quantity (MT)", min_value=0.0, value=0.0, step=0.1)
    total_inv_tagged_qty = 0
    total_paper_buy_tagged_qty = 0
    updated_invoice = pd.DataFrame()
    updated_paper = pd.DataFrame()
    
    for inv_selection in selected_invoice:
        inv_id = inv_selection.split(" - ")[0]
        invoice_id = invoice_df[invoice_df['filter'] == inv_id]['_id'].values[0] if inv_id else None
        if invoice_id:
            invoice = invoice_df[invoice_df['_id'] == invoice_id].iloc[0]
            updated_invoice = pd.concat([updated_invoice, pd.DataFrame([invoice])], ignore_index=True)   
            total_inv_tagged_qty = total_inv_tagged_qty + float(invoice['open_qty'])
            total_inv_tagged_qty = min(total_inv_tagged_qty, hedge_qty)
    
    for paper_buy in selected_buys:
        pap_id = paper_buy.split(" - ")[0]
        paper_id = available_buys[available_buys['filter'] == pap_id]['position_id'].values[0] if pap_id else None
        if paper_id:
            paper = available_buys[available_buys['position_id'] == paper_id].iloc[0]
            updated_paper = pd.concat([updated_paper, pd.DataFrame([paper])], ignore_index=True)
            total_paper_buy_tagged_qty = total_paper_buy_tagged_qty + float(paper['invoice_open_quantity'])
            total_paper_buy_tagged_qty = min(total_paper_buy_tagged_qty, hedge_qty)
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown(f"**Total Invoices Quantity**: {total_inv_tagged_qty}")
    with col2:
        st.markdown(f"**Total Paper buy Quantity**: {total_paper_buy_tagged_qty}")

    tagged_qty = min(total_inv_tagged_qty, total_paper_buy_tagged_qty)
    st.markdown(f"**Total Tagged Quantity**: {tagged_qty}")
    
    left_tagging_invoice = tagged_qty 
    left_tagging_paper = tagged_qty

    info_updated_invoice = updated_invoice.copy()
    info_updated_paper = updated_paper.copy()

    for i in range(len(info_updated_invoice)):
        current_qty = float(info_updated_invoice.iloc[i]['open_qty'])
        
        if current_qty >= left_tagging_invoice:
            info_updated_invoice.at[i, 'open_qty'] = current_qty - left_tagging_invoice
            left_tagging_invoice = 0
            break  # stop once tagging is done
        else:
            info_updated_invoice.at[i, 'open_qty'] = 0
            left_tagging_invoice -= current_qty  # subtract the used quantity

    for i in range(len(info_updated_paper)):
        current_qty = float(info_updated_paper.iloc[i]['invoice_open_quantity'])

        if current_qty >= left_tagging_paper:
            info_updated_paper.at[i, 'invoice_open_quantity'] = current_qty - left_tagging_paper
            left_tagging_paper = 0
            break  # stop once tagging is done  
        else:
            info_updated_paper.at[i, 'invoice_open_quantity'] = 0
            left_tagging_paper -= current_qty  # subtract the used quantity

    if not updated_invoice.empty:
        st.table(info_updated_invoice[['invoicenumber','invoicedate','product','item_name','item_quantity','item_unit','item_unitrate','open_qty']])
    if not updated_paper.empty:
        st.table(info_updated_paper[['position_id','product','artefact_status','customer_name','invoice_open_quantity','price_rate','expiry_date']])

    tag_button = st.button('Tag Invoice to Selected Paper Buy Entries')

    if tag_button:
        if info_updated_invoice.empty:
            st.info("Invoice Not selected")
        elif info_updated_paper.empty:
            st.info("Papaer Buy not selelcted")
        
        elif info_updated_invoice.empty and info_updated_paper.empty:
            st.info("Invoice and Paper Buy not selected")
    
        else:
            fifo_matches, buy_with_remaining, sell_with_remaining = fifo_match_invoice(updated_paper, updated_invoice)
            buy_with_remaining = buy_with_remaining[buy_with_remaining['remaining_qty'] > 0]
            sell_with_remaining = sell_with_remaining[sell_with_remaining['remaining_qty'] > 0]
            if not fifo_matches.empty:
                utils.create_paper_invoice_tagging(fifo_matches)
            if not buy_with_remaining.empty:
                utils.update_paper_buy_invoice_df(buy_with_remaining)

            st.rerun()
    st.markdown("---")
    # Show tagged invoices
    st.subheader("Tagged Invoices")
    if invoice_tagged_zata_df.empty:
        st.info("No invoices tagged yet")
    else:
        st.dataframe(invoice_tagged_zata_df[['position_id','invoicenumber','product','tagged_qty']],hide_index=True)


# --- UI Components ---
def render_bill_tagging_tab(q1,q2,q3):
    """Render the bill tagging tab"""
    st.header("📋 Bill Tagging")

    bill_df = utils.get_question_data(q1)
    paper_sell_df = utils.get_table_data(q2)
    bill_tagged_zata_df = utils.get_table_data(q3)
    paper_sell_df = paper_sell_df[(paper_sell_df['bill_open_quantity'] > 0) & (paper_sell_df['artefact_status'] == 'OPEN')]
    
    if bill_df.empty:
        st.warning("No Billss available. Please upload Billss from OMS system.")
        return
    
    # =============================================================================
    # COMMODITY FILTER FOR BillsS
    # =============================================================================
    # Get all unique commodities from Billss
    all_bill_commodities = set(list(bill_df['product']))
    
    if all_bill_commodities:
        # Add "All Commodities" option
        commodity_options = ["All Commodities"] + sorted(list(all_bill_commodities))
        
        # Create commodity filter
        col1,col2 = st.columns([1,3])
        with col1:
            selected_commodity = st.selectbox(
                "Filter Billss by Commodity:",
                options=commodity_options,
                index=0,
                key="bill_commodity_filter"
            )
        with col2:
            # Show filter status
            if selected_commodity != "All Commodities":
                st.info(f"📊 Showing Bills for: **{selected_commodity}**")
            else:
                commodities_str = ", ".join(sorted(all_bill_commodities))
                st.info(f"📊 Showing Billss for: **All Commodities** ({commodities_str})")
    else:
        selected_commodity = "All Commodities"
        st.info("No Billss available to filter")
    
    st.markdown("---")
        
    # Bills selection (filtered by commodity)
    if selected_commodity != "All Commodities":
        # Filter Billss by selected commodity
        bill_df = bill_df[bill_df['product'] == selected_commodity]
        paper_sell_df = paper_sell_df[paper_sell_df['product'] == selected_commodity]
        if bill_df.empty:
            st.warning(f"No {selected_commodity} Billss available.")
            return

    # Create filtered Bills options

    available_sells = []
    available_sells = paper_sell_df[paper_sell_df['bill_open_quantity'] > 0]
    bill_df['filter'] = bill_df['billnumber'].astype(str) + ' with open qty ' + bill_df['open_qty'].astype(str) + ' ' + bill_df['item_unit'] +' '+ bill_df['product']
    available_sells['filter'] = available_sells['position_id'] + ' with open qty ' + available_sells['bill_open_quantity'].astype(str) +' MT '+ available_sells['product']
    col1,col2 = st.columns([1,1])
    bill_df_info = bill_df.copy()
    available_sells_info = available_sells.copy()
    bill_df_info = bill_df_info[bill_df_info['open_qty'] != 0]
    available_sells_info = available_sells_info[available_sells_info['bill_open_quantity'] != 0]
    # bill_df_info = bill_df_info[bill_df_info['supplier_name'].isin(available_sells_info['supplier_group_name'].unique())]
    # available_sells_info = available_sells_info[available_sells_info['supplier_group_name'].isin(bill_df_info['supplier_name'].unique())]
    # Find the common supplier names
    common_suppliers = set(bill_df_info['supplier_name']).intersection(
        available_sells_info['supplier_group_name']
    )

    # Filter both using the common set
    # bill_df_info = bill_df_info[bill_df_info['supplier_name'].isin(common_suppliers)]
    # available_sells_info = available_sells_info[available_sells_info['supplier_group_name'].isin(common_suppliers)]

    with col1:
        st.subheader("Available Bills")
        if not bill_df_info.empty:
            col11, col22, col33 = st.columns([1, 1, 1])
            with col11:
                bill_number_filter = st.multiselect(
                    "Filter Bills by Bill Number",
                    sorted(bill_df_info['billnumber'].unique()),
                    key="bill_number_filter"
                )
            with col22:
                bill_supplier_filter = st.multiselect(
                    "Filter Bills by Supplier Name",
                    sorted(bill_df_info['supplier_name'].unique()),
                    key="bill_supplier_filter"
                )
            with col33:
                # Convert billdate to datetime first
                bill_df_info['billdate'] = pd.to_datetime(bill_df_info['billdate'], errors='coerce').dt.date

                start_date, end_date = st.date_input(
                    "Filter Bills by Date Range",
                    [bill_df_info['billdate'].min(), bill_df_info['billdate'].max()],
                    key="bill_date_filter"
                )
            if bill_number_filter:
                bill_df_info = bill_df_info[bill_df_info['billnumber'].isin(bill_number_filter)]
            if bill_supplier_filter:
                bill_df_info = bill_df_info[bill_df_info['supplier_name'].isin(bill_supplier_filter)]
            if start_date and end_date:
                bill_df_info = bill_df_info[
                    (bill_df_info['billdate'] >= start_date) &
                    (bill_df_info['billdate'] <= end_date)
                ]
    
    with col2:
        if not available_sells.empty:
            st.subheader("Available Paper Sell positions")
            col11,col22,col33 = st.columns([1,1,1])
            with col11:
                paper_sell_number_filter = st.multiselect(
                    "Filter paper sell by position_id",
                    sorted(available_sells_info['position_id'].unique()),
                    key="sell_position_id_filter"
                )
            with col22:
                paper_sell_supplier_filter = st.multiselect(
                    "Filter paper sell by Supplier Name",
                    sorted(available_sells_info['supplier_name'].unique()),
                    key="paper_sell_supplier_filter"
                )
            with col33:
                available_sells_info['transaction_date'] = pd.to_datetime(
                    available_sells_info['transaction_date'], errors='coerce'
                )
                if available_sells_info['transaction_date'].notna().any():
                    min_date = available_sells_info['transaction_date'].min().date()
                    max_date = available_sells_info['transaction_date'].max().date()
                    start_date, end_date = st.date_input(
                        "Filter Paper Sell by Date Range",
                        [min_date, max_date],
                        key="paper_sell_date_filter"
                    )
                else:
                    start_date, end_date = None, None
            if paper_sell_number_filter:
                available_sells_info = available_sells_info[available_sells_info['transaction_date'].isin(paper_sell_number_filter)]
            if paper_sell_supplier_filter:
                available_sells_info = available_sells_info[available_sells_info['transaction_date'].isin(paper_sell_supplier_filter)]
            if start_date and end_date:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                available_sells_info = available_sells_info[
                    (available_sells_info['transaction_date'] >= start_date) &
                    (available_sells_info['transaction_date'] <= end_date)
                ]
    
    with col1:
        st.dataframe(bill_df_info[['billnumber','billdate','supplier_name', 'product','item_name','open_qty','item_unit','open_qty_value']], hide_index=True) 
    
    with col2:
        st.dataframe(available_sells_info[['position_id','product', 'artefact_status','supplier_name','bill_open_quantity','price_rate','expiry_date']],hide_index=True)
    
    col1,col2 = st.columns([1,1])
    with col1:
        selected_bill = st.multiselect("Select Bills to Tag", list(bill_df['filter'].unique()), key="bills_select")
        formatted_text = "\n\n".join(str(item) for item in selected_bill) 
        st.markdown(f"{formatted_text}")

    with col2:
        selected_sells = st.multiselect("Select Paper Sell Entries to Tag (Multiple selection allowed)", list(available_sells['filter'].unique()))
        formatted_text_sell = "\n\n".join(str(item) for item in selected_sells) 
        st.markdown(f"{formatted_text_sell}")
    
    with col1:
        hedge_bill_qty = st.number_input(
            "Enter Quantity (MT)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="hedge_bill_qty_paper_sell"
        )

    
    total_bill_tagged_qty = 0
    total_paper_sell_tagged_qty = 0
    updated_bill = pd.DataFrame()
    updated_paper = pd.DataFrame()
    
    for bill_selection in selected_bill:
        bil_id = bill_selection.split(" - ")[0]
        bill_id = bill_df[bill_df['filter'] == bil_id]['_id'].values[0] if bil_id else None
        if bill_id:
            bills = bill_df[bill_df['_id'] == bill_id].iloc[0]
            updated_bill = pd.concat([updated_bill, pd.DataFrame([bills])], ignore_index=True)   
            total_bill_tagged_qty = total_bill_tagged_qty + float(bills['open_qty'])
            total_bill_tagged_qty = min(total_bill_tagged_qty, hedge_bill_qty)
    
    for paper_sell in selected_sells:
        pap_id = paper_sell.split(" - ")[0]
        paper_id = available_sells[available_sells['filter'] == pap_id]['position_id'].values[0] if pap_id else None
        if paper_id:
            paper = available_sells[available_sells['position_id'] == paper_id].iloc[0]
            updated_paper = pd.concat([updated_paper, pd.DataFrame([paper])], ignore_index=True)
            total_paper_sell_tagged_qty = total_paper_sell_tagged_qty + float(paper['bill_open_quantity'])
            total_paper_sell_tagged_qty = min(total_paper_sell_tagged_qty, hedge_bill_qty)
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown(f"**Total Billss Quantity**: {total_bill_tagged_qty}")
    with col2:
        st.markdown(f"**Total Paper sell Quantity**: {total_paper_sell_tagged_qty}")

    tagged_qty = min(total_bill_tagged_qty, total_paper_sell_tagged_qty)
    st.markdown(f"**Total Tagged Quantity**: {tagged_qty}")
    
    left_tagging_bill = tagged_qty 
    left_tagging_paper = tagged_qty

    info_updated_bill = updated_bill.copy()
    info_updated_paper = updated_paper.copy()

    for i in range(len(info_updated_bill)):
        current_qty = float(info_updated_bill.iloc[i]['open_qty'])
        
        if current_qty >= left_tagging_bill:
            info_updated_bill.at[i, 'open_qty'] = current_qty - left_tagging_bill
            left_tagging_bill = 0
            break  # stop once tagging is done
        else:
            info_updated_bill.at[i, 'open_qty'] = 0
            left_tagging_bill -= current_qty  # subtract the used quantity

    for i in range(len(info_updated_paper)):
        current_qty = float(info_updated_paper.iloc[i]['bill_open_quantity'])

        if current_qty >= left_tagging_paper:
            info_updated_paper.at[i, 'bill_open_quantity'] = current_qty - left_tagging_paper
            left_tagging_paper = 0
            break  # stop once tagging is done  
        else:
            info_updated_paper.at[i, 'bill_open_quantity'] = 0
            left_tagging_paper -= current_qty  # subtract the used quantity

    if not updated_bill.empty:
        st.table(info_updated_bill[['billnumber','billdate','product','item_name','item_quantity','item_unit','item_unitrate','open_qty']])
    if not updated_paper.empty:
        st.table(info_updated_paper[['position_id','product','artefact_status','supplier_name','bill_open_quantity','price_rate','expiry_date']])

    tag_button = st.button('Tag Bill to Selected Paper Sell Entries')

    if tag_button:
        if info_updated_bill.empty:
            st.info("Bill Not selected")
        elif info_updated_paper.empty:
            st.info("Papaer Sell not selelcted")
        
        elif info_updated_bill.empty and info_updated_paper.empty:
            st.info("Bill and Paper Sell not selected")
    
        else:
            fifo_matches, buy_with_remaining, sell_with_remaining = fifo_match_bill(updated_paper, updated_bill)
            buy_with_remaining = buy_with_remaining[buy_with_remaining['remaining_qty'] > 0]
            sell_with_remaining = sell_with_remaining[sell_with_remaining['remaining_qty'] > 0]
            if not fifo_matches.empty:
                utils.create_paper_bill_tagging(fifo_matches)
            if not buy_with_remaining.empty:
                utils.update_paper_sell_bill_df(buy_with_remaining, 'remaining_qty', 'position_id')

            st.rerun()
    st.markdown("---")
    # Show tagged Billss
    st.subheader("Tagged Billss")
    if bill_tagged_zata_df.empty:
        st.info("No Billss tagged yet")
    else:
        st.dataframe(bill_tagged_zata_df,hide_index=True)

def render_hedging_tab():
    """Render the main hedging tab with sub-tabs"""
    # st.header("📊 Hedging Management")
    st.markdown("<h1 class='centered-title'>📊 Paper Postion Management</h1>", unsafe_allow_html=True)

    
    # Initialize active tab in session state if not present
    if 'active_hedge_tab' not in st.session_state:
        st.session_state.active_hedge_tab = "🔄 Manage Positions"  # Default to Manage Positions tab
    
    # Create sub-tabs for hedging operations
    hedge_tab1, hedge_tab2, hedge_tab3, hedge_tab4, hedge_tab5 = st.tabs([
        "➕ Create Paper Sell", 
        "➕ Create Paper Buy", 
        "🔄 Manage Positions",
        "🔄 Roll-on Positions",
        "📈 Position Dashboard"
    ])
    
    with hedge_tab1:
        render_create_paper_sell(constants_variable.MCX_PRODUCTS)

    with hedge_tab2:
        render_create_paper_buy(constants_variable.MCX_PRODUCTS)
        
    with hedge_tab3:
        render_manage_positions("paper_sell_entry","paper_buy_entry")

    with hedge_tab4:
        render_rollover_positions("paper_sell_entry","paper_buy_entry")
        # render_position_dashboard("paper_sell_entry","paper_buy_entry","hedge_paper_settle_entry")

def render_create_paper_sell(MCX_PRODUCTS):
    """Render the create paper sell sub-tab"""
    st.subheader("Create New Paper Sell Entry")
    
    # Initialize session state for real-time updates and form inputs
    if 'ps_num_lots' not in st.session_state:
        st.session_state.ps_num_lots = 1
    if 'ps_lot_size' not in st.session_state:
        st.session_state.ps_lot_size = 25.0
    if 'ps_supplier_input' not in st.session_state:
        st.session_state.ps_supplier_input = ""
    # Note: No session state for product_select to avoid index issues
    if 'ps_lots_input' not in st.session_state:
        st.session_state.ps_lots_input = 1
    if 'ps_lot_size_input' not in st.session_state:
        st.session_state.ps_lot_size_input = 25.0
    if 'ps_price_input' not in st.session_state:
        st.session_state.ps_price_input = 800.0
    if 'ps_expiry_input' not in st.session_state:
        st.session_state.ps_expiry_input = date.today() + timedelta(days=30)
    if 'ps_transaction_input' not in st.session_state:
        st.session_state.ps_transaction_input = date.today()
    # Note: No session state for exchange_select to avoid index issues
    
    # Check if form was successfully submitted and reset fields
    if st.session_state.get('ps_form_success', False):
        st.session_state.ps_supplier_input = ""
        st.session_state.ps_lots_input = 1
        st.session_state.ps_lot_size_input = 25.0
        st.session_state.ps_price_input = 800.0
        st.session_state.ps_expiry_input = date.today() + timedelta(days=30)
        st.session_state.ps_transaction_input = date.today()
        st.session_state.ps_form_success = False  # Reset the flag
    
    col1, col2 = st.columns(2)

    with col1:
        # Product selection
        product = st.selectbox("Product", MCX_PRODUCTS, key="ps_product_select")

        # Supplier name (auto-capitalized)
        supplier_name_df = utils.get_question_data(8959)
        supplier_name_list = supplier_name_df['suppliername'].dropna().unique().tolist()
        supplier_name = st.selectbox(
                        "Supplier Name",
                        options=supplier_name_list,
                        # index=None,  # So nothing is preselected
                        index = 0,
                        placeholder="Enter supplier name",
                        key="ps_supplier_input"
                    )
        if supplier_name:
            supplier_name = supplier_name.upper()

        # Number of lots - use session state for real-time updates
        num_lots = st.number_input("Number of Lots", min_value=1, step=1, key="ps_lots_input")
        # Expiry date
        expiry_date = st.date_input("Expiry Date", key="ps_expiry_input")     
                
    with col2:
        # Price/Rate
        price_rate = st.number_input("Price/Rate", min_value=0.0, step=0.01, key="ps_price_input")

        # suppliergroupname
        result = supplier_name_df.loc[supplier_name_df['suppliername'] == supplier_name, 'suppliergroupname'].values
        supplier_group_name = result[0] if len(result) > 0 else ""
        st.text_input("Supplier Group Name", value=supplier_group_name, key="ps_supplier_group_input")

        # Lot size - use session state for real-time updates
        lot_size = st.number_input("Lot Size (MT)", min_value=0.1, step=0.1, key="ps_lot_size_input")

        # Transaction date
        transaction_date = st.date_input("Date of Transaction", key="ps_transaction_input")

    with col1:
        col3, col4 = st.columns([1,3])
        with col3:
            # Calculate total quantity in real-time using current input values
            total_quantity = num_lots * lot_size
            st.markdown(
                f"""
                <div style="font-size:16px; font-weight:600;">Total Quantity (MT)</div>
                <div style="font-size:20px; color:#1E90FF;">{total_quantity:.1f}</div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            # Show calculation breakdown
            st.info(f"📊 Calculation: {num_lots} lots × {lot_size} MT = {total_quantity:.1f} MT")

        # Exchange selection (default to MCX for now)
        exchange = st.selectbox("Exchange", EXCHANGES, key="ps_exchange_select")  # Remove index parameter to get actual string values
    
    with col2:
        # Safety check: ensure exchange is a valid string before lookup
        if isinstance(exchange, str) and exchange in EXCHANGE_CURRENCIES:
                    currency = EXCHANGE_CURRENCIES[exchange]
                    st.info(f"Currency: {currency}")
        else:
            st.error(f"❌ Invalid exchange value: {exchange} (type: {type(exchange)})")
            st.info("Please select a valid exchange from the dropdown")
            return

        # Conversion rate (if needed for LME)
        if exchange == 'LME':
            conversion_rate = st.number_input("Currency Conversion Rate (to INR)", min_value=0.0, step=0.01, key="ps_conversion_input")
        else:
            conversion_rate = 1.0

    # Initialize form reset counter for automatic form clearing
    if 'ps_form_reset_counter' not in st.session_state:
        st.session_state.ps_form_reset_counter = 0


    # Create form for submission with dynamic key for automatic reset
    with st.form(f"paper_sell_form_{st.session_state.ps_form_reset_counter}"):
        submitted = st.form_submit_button("Create Paper Sell Entry")
        if submitted and supplier_name:
            # Always create a new position for new entries
            # position_id = f"PS_{product}_{supplier_name}_{random.randint(100000, 999999)}"\
            timestamp = int(time.time() * 1000)  # milliseconds since epoch
            rand_part = random.randint(1000, 9999)
            unique_number = f"{timestamp}{rand_part}"[-10:]  # keep last 10 digits for consistency

            # Final position ID
            position_id = f"HEDGE_{unique_number}"
            paper_sell_prev_df = utils.get_table_data("paper_sell_entry")  # Refresh cached data
            existing_entity_ids = paper_sell_prev_df['entry_id'].unique().tolist()
            entry_id = generate_entity_id(existing_entity_ids)

            # Create entry data
            entry_data = {
            "position_id": position_id,
            "entry_id": entry_id,
            "product": product,
            "status": "OPEN",
            "supplier_name": supplier_name,
            "supplier_group_name": supplier_group_name,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "total_quantity": total_quantity,
            "paper_open_quantity": total_quantity,
            "bill_open_quantity": total_quantity,
            "price_rate": price_rate,
            "expiry_date": expiry_date,
            "transaction_date": transaction_date,
            "exchange": exchange,
            "currency": currency,
            "conversion_rate": conversion_rate,
            "roll_over_quantity": 0.0,
            }

            # Create a DataFrame
            paper_sell_entry_df = pd.DataFrame([entry_data])
            st.success(f"✅ Paper Sell Entry created successfully!")
            st.info(f"Position ID: {position_id}\nEntry ID: {entry_id}\nTotal Quantity: {total_quantity} MT")
            utils.create_paper_sell_entry_push(paper_sell_entry_df)

            # Set success flag to trigger form reset
            st.session_state.ps_form_success = True
            st.info("Form will reset on next page load. You can create another entry above.")
        elif submitted and not supplier_name:
            st.error("Please enter a supplier name")

def render_create_paper_buy(MCX_PRODUCTS):
    """Render the create paper buy sub-tab"""
    st.subheader("Create New Paper Buy Entry")
    
    # Initialize session state for real-time updates and form inputs
    if 'pb_num_lots' not in st.session_state:
        st.session_state.pb_num_lots = 1
    if 'pb_lot_size' not in st.session_state:
        st.session_state.pb_lot_size = 25.0
    if 'pb_customer_input' not in st.session_state:
        st.session_state.pb_customer_input = ""
    # Note: No session state for product_select to avoid index issues
    if 'pb_lots_input' not in st.session_state:
        st.session_state.pb_lots_input = 1
    if 'pb_lot_size_input' not in st.session_state:
        st.session_state.pb_lot_size_input = 25.0
    if 'pb_price_input' not in st.session_state:
        st.session_state.pb_price_input = 800.0
    if 'pb_expiry_input' not in st.session_state:
        st.session_state.pb_expiry_input = date.today() + timedelta(days=30)
    if 'pb_transaction_input' not in st.session_state:
        st.session_state.pb_transaction_input = date.today()
    # Note: No session state for exchange_select to avoid index issues
    
    # Check if form was successfully submitted and reset fields
    if st.session_state.get('pb_form_success', False):
        st.session_state.pb_customer_input = ""
        st.session_state.pb_lots_input = 1
        st.session_state.pb_lot_size_input = 25.0
        st.session_state.pb_price_input = 800.0
        st.session_state.pb_expiry_input = date.today() + timedelta(days=30)
        st.session_state.pb_transaction_input = date.today()
        st.session_state.pb_form_success = False  # Reset the flag
    col1, col2 = st.columns(2)

    with col1:
        # Product selection
        product = st.selectbox("Product", MCX_PRODUCTS, key="pb_product_select")

        # Customer name (auto-capitalized)
        customer_name_df = utils.get_question_data(8960)
        customer_name_list = customer_name_df['customername'].dropna().unique().tolist()
        customer_name = st.selectbox(
                        "Customer Name",
                        options=customer_name_list,
                        index=0,  # So nothing is preselected
                        placeholder="Enter customer name",
                        key="pb_customer_input"
                    )
        if customer_name:
            customer_name = customer_name.upper()

        # Number of lots - use session state for real-time updates
        num_lots = st.number_input("Number of Lots", min_value=1, step=1, key="pb_lots_input")

        # Expiry date
        expiry_date = st.date_input("Expiry Date", key="pb_expiry_input")
                
    with col2:
        # Price/Rate
        price_rate = st.number_input("Price/Rate", min_value=0.0, step=0.01, key="pb_price_input")

        # cudtomer group name 
        result = customer_name_df.loc[customer_name_df['customername'] == customer_name, 'customergroupname'].values
        customer_group_name = result[0] if len(result) > 0 else ""
        st.text_input("Customer Group Name", value=customer_group_name, key="pb_customer_group_input")

        # Lot size - use session state for real-time updates
        lot_size = st.number_input("Lot Size (MT)", min_value=0.1, step=0.1, key="pb_lot_size_input")

        # Transaction date
        transaction_date = st.date_input("Date of Transaction", key="pb_transaction_input")

    with col1:
        col3, col4 = st.columns([1,3])
        with col3:
            # Calculate total quantity in real-time using current input values
            total_quantity = num_lots * lot_size
            st.markdown(
                f"""
                <div style="font-size:16px; font-weight:600;">Total Quantity (MT)</div>
                <div style="font-size:20px; color:#1E90FF;">{total_quantity:.1f}</div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            # Show calculation breakdown
            st.info(f"📊 Calculation: {num_lots} lots × {lot_size} MT = {total_quantity:.1f} MT")
    
        # Exchange selection (default to MCX for now)
        exchange = st.selectbox("Exchange", EXCHANGES, key="pb_exchange_select")  # Remove index parameter to get actual string values

    with col2:
        # Safety check: ensure exchange is a valid string before lookup
        if isinstance(exchange, str) and exchange in EXCHANGE_CURRENCIES:
                    currency = EXCHANGE_CURRENCIES[exchange]
                    st.info(f"Currency: {currency}")
        else:
            st.error(f"❌ Invalid exchange value: {exchange} (type: {type(exchange)})")
            st.info("Please select a valid exchange from the dropdown")
            return

        # Conversion rate (if needed for LME)
        if exchange == 'LME':
            conversion_rate = st.number_input("Currency Conversion Rate (to INR)", min_value=0.0, step=0.01, key="pb_conversion_input")
        else:
            conversion_rate = 1.0

    # Initialize form reset counter for automatic form clearing
    if 'pb_form_reset_counter' not in st.session_state:
        st.session_state.pb_form_reset_counter = 0

    # Create form for submission with dynamic key for automatic reset
    with st.form(f"paper_buy_form_{st.session_state.pb_form_reset_counter}"):
        submitted = st.form_submit_button("Create Paper Buy Entry")
        
        if submitted and customer_name:
            # Always create a new position for new entries
            # position_id = f"PB_{product}_{customer_name}_{random.randint(100000, 999999)}"
            timestamp = int(time.time() * 1000)  # milliseconds since epoch
            rand_part = random.randint(1000, 9999)
            unique_number = f"{timestamp}{rand_part}"[-10:]  # keep last 10 digits for consistency

            # Final position ID
            position_id = f"HEDGE_{unique_number}"
            # Create new position

            paper_buy_prev_df = utils.get_table_data("paper_buy_entry")  # Refresh cached data
            existing_entity_ids = paper_buy_prev_df['entry_id'].unique().tolist()
            entry_id = generate_entity_id(existing_entity_ids)
            # # Add paper buy entry
            # entry_id = hedging_module.add_paper_buy(
            #     position_id=position_id,
            #     quantity=total_quantity,
            #     exchange=exchange,
            #     rate=price_rate,
            #     currency=currency,
            #     conversion_rate=conversion_rate,
            #     customer_name=customer_name,
            #     num_lots=num_lots,
            #     lot_size=lot_size,
            #     transaction_date=transaction_date,
            #     due_date=expiry_date
            # )
            
            # Create entry data
            entry_data = {
            "position_id": position_id,
            "entry_id": entry_id,
            "product": product,
            "status": "OPEN",
            "customer_name": customer_name,
            "customer_group_name": customer_group_name,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "total_quantity": total_quantity,
            "paper_open_quantity": total_quantity,
            "invoice_open_quantity": total_quantity,
            "price_rate": price_rate,
            "expiry_date": expiry_date,
            "transaction_date": transaction_date,
            "exchange": exchange,
            "currency": currency,
            "conversion_rate": conversion_rate,
            "roll_over_quantity": 0.0,
            }
            # Create a DataFrame
            paper_buy_entry_df = pd.DataFrame([entry_data])

            utils.create_paper_buy_entry_push(paper_buy_entry_df)
            st.success(f"✅ Paper Buy Entry created successfully!")
            st.info(f"Position ID: {position_id}\nEntry ID: {entry_id}\nTotal Quantity: {total_quantity} MT")

            # Set success flag to trigger form reset
            st.session_state.pb_form_success = True
            st.info("🔄 Form will reset on next page load. You can create another entry above.")
        elif submitted and not customer_name:
            st.error("Please enter a customer name")

def render_manage_positions(q1,q2):
    """Render the manage positions sub-tab"""
    col1, col2 = st.columns([20,1])
    with col1:
        st.subheader("Manage Hedging Positions")
    with col2:
        st.button("ᯓ➤", help="Manage your paper sell and buy positions here. Use the filters to narrow down positions by commodity. You can view open and closed positions separately. Select positions to initiate hedging or Roll over.")   
    # Fetch paper sell and buy data for display
    paper_sell_df = utils.get_table_data(q1)
    paper_buy_df = utils.get_table_data(q2)

    paper_sell_df_copy = paper_sell_df.copy()
    paper_buy_df_copy = paper_buy_df.copy()
    paper_sell_df_copy = paper_sell_df_copy.drop(columns=['supplier_name', 'supplier_group_name'], errors='ignore')
    paper_buy_df_copy = paper_buy_df_copy.drop(columns=['customer_name','customer_group_name'], errors='ignore')
    comb_df_copy = pd.concat([paper_sell_df_copy, paper_buy_df_copy], ignore_index=True)
    comb_df = comb_df_copy[comb_df_copy['status'] == 'OPEN']
    comb_df_closed = comb_df_copy[comb_df_copy['status'] == 'CLOSED']

    
    # =============================================================================
    # COMMODITY FILTER
    # =============================================================================
    # Get all unique commodities from positions
    all_commodities = set()
    for index, row in comb_df.iterrows():
        if not row['product']:
            continue
        all_commodities.add(row['product'])

    if all_commodities:
        # Add "All Commodities" option
        commodity_options = ["All Commodities"] + sorted(list(all_commodities))
        
        col1, col2 = st.columns([1,3])
        with col1:
            # Create commodity filter
            selected_commodity = st.selectbox(
                "Filter by Commodity:",
                options=commodity_options,
                index=0,
                key="commodity_filter"
            )
        with col2:
            # Show filter status
            if selected_commodity != "All Commodities":
                st.info(f"📊 Showing positions for: **{selected_commodity}**")
            else:
                commodities_str = ", ".join(sorted(all_commodities))
                st.info(f"📊 Showing positions for: **All Commodities** ({commodities_str})")
    else:
        selected_commodity = "All Commodities"
        st.info("No positions available to filter")
    
    st.markdown("---")
    
    if comb_df.empty:
        st.info("No hedging positions created yet")
        return
        
    # not required
    # =============================================================================
    # SEGREGATE BUY AND SELL POSITIONS (WITH COMMODITY FILTER)
    # =============================================================================
    # Separate positions by type for proper hedging workflow
    # Buy positions can only be hedged with Sell positions (and vice versa)
    buy_positions = []
    sell_positions = []
    closed_positions = []
    
    # for position_id, position in hedging_module.positions.items():
    # Apply commodity filter
        
    if not comb_df_closed.empty:
        for index, row in comb_df_closed.iterrows():
            closed_positions.append((row['position_id'], row['entry_id']))
    if not comb_df.empty:
        for index, row in paper_sell_df[paper_sell_df['status'] == 'OPEN'].iterrows():
            sell_positions.append((row['position_id'], row['entry_id']))
        for index, row in paper_buy_df[paper_buy_df['status'] == 'OPEN'].iterrows():
            buy_positions.append((row['position_id'], row['entry_id']))
    buy_positions = list(set(buy_positions))
    sell_positions = list(set(sell_positions))
    closed_positions = list(set(closed_positions))
     
    
    # =============================================================================
    # HEDGING WORKFLOW SUMMARY
    # =============================================================================
    if buy_positions or sell_positions:
        # Show commodity-specific or general workflow info
        commodity_text = f" for {selected_commodity}" if selected_commodity != "All Commodities" else ""
        # Show summary of available positions for hedging
        if buy_positions and sell_positions:
            st.success(f"✅ **Hedging Available{commodity_text}**: {len(buy_positions)} buy position(s) can be hedged with {len(sell_positions)} sell position(s)")
        elif buy_positions:
            st.warning(f"⚠️ **Need Sell Positions{commodity_text}**: {len(buy_positions)} buy position(s) waiting for sell positions to hedge with")
        elif sell_positions:
            st.warning(f"⚠️ **Need Buy Positions{commodity_text}**: {len(sell_positions)} sell position(s) waiting for buy positions to hedge with")

    # =============================================================================
    # PAPER BUY POSITIONS SECTION
    # =============================================================================
    if buy_positions:
        commodity_text = f" ({selected_commodity})" if selected_commodity != "All Commodities" else ""
        st.subheader(f"📈 Paper Buy Positions{commodity_text}")
        
        for position_id, entry_id in buy_positions:
            # Find the row by entry_id
            row_df = paper_buy_df[(paper_buy_df['status'] == 'OPEN') & (paper_buy_df['entry_id'] == entry_id) & (paper_buy_df['position_id'] == position_id)]
            if not row_df.empty:
                row = row_df.iloc[0]

                # Fetch details safely
                commodity = row.get("product", "N/A")
                paper_open_qty = row.get("paper_open_quantity", 0)
                type_ = 'BUY'
                quantity = row.get("total_quantity", 0)
                exchange = row.get("exchange", "N/A")
                rate = row.get("price_rate", 0)
                timestamp = row.get("timestamp", pd.Timestamp.now())
                if pd.notnull(timestamp):
                    timestamp = pd.to_datetime(timestamp).strftime('%d %b %Y')
                transaction_date = row.get("transaction_date", "N/A")
                due_date = row.get("expiry_date", "N/A")
                customer_name = row.get("customer_name", "")
                customer_group_name = row.get("customer_group_name", "")
                num_lots = row.get("num_lots", None)
                lot_size = row.get("lot_size", None)
                supplier_name = row.get("supplier_name", "")
                supplier_group_name = row.get("supplier_group_name", "")

                 # Display expander for each buy position
                with st.expander(f"🟢 Buy Position {position_id} - {commodity} ({paper_open_qty:.1f} MT open)", expanded=True):
                    col1, col2, col3, col4,col5,col6 = st.columns([1,1,1,1,1,3])
                    
                    with col1:
                        st.metric("Latest Action", type_.upper())
                    with col2:
                        st.metric("Open Quantity", f"{paper_open_qty:.1f} MT")
                    with col3:
                        st.metric("Total Quantity", f"{quantity:.1f} MT")
                    with col4:
                        st.metric("Exchange", exchange)
                    with col5:
                        st.metric("Rate", f"{rate}")
                    with col6:
                        st.metric("customer", customer_group_name)
                    
                    # =============================================================================
                    # HEDGING FUNCTIONALITY FOR BUY POSITIONS
                    # =============================================================================
                    st.subheader("💰 Settle This Buy Position")
                    # Get all available SELL positions for hedging (cross-position hedging only)
                    # Filter sell positions by commodity to match the current buy position
                    matching_sell_positions = paper_sell_df[(paper_sell_df['product'] == commodity) & (paper_sell_df['status'] == 'OPEN')][['position_id', 'entry_id']]
                    
                    if not matching_sell_positions.empty:
                        # st.write(f"**Available {commodity} Paper Sell Positions to Settle With:**")
                        
                        # Use simple string-based selection to avoid type issues
                        if not matching_sell_positions.empty:
                            # Create simple string options that include the position ID
                            sell_options = [str(position_id) for position_id in matching_sell_positions['position_id'].tolist()]
                            col1 , col2, col3 = st.columns([1,1.5,1.5])
                            with col1:
                                selected_sell_string = st.selectbox(
                                    "Select Paper Sell Position to Settle With", 
                                    options=sell_options,  # Use position IDs as strings
                                    key=f"sell_select_{position_id}"
                                )
                            # Parse the selected position ID and get current quantities
                            if selected_sell_string:
                                selected_sell_id = selected_sell_string
                                # Find the corresponding position object
                                selected_sell_position  = matching_sell_positions[matching_sell_positions['position_id'] == selected_sell_id]['entry_id']
                                if not selected_sell_position.empty:
                                    # Auto-populate supplier name from the selected sell position
                                    selected_sell_position_value = selected_sell_position.iloc[0]
                                    latest_sell_entry = paper_sell_df[paper_sell_df['entry_id'] == str(selected_sell_position_value)]
                                    selected_sell_qty = latest_sell_entry['paper_open_quantity'].iloc[0] if not latest_sell_entry.empty else 0
                                    if not latest_sell_entry.empty:
                                        supplier_name = latest_sell_entry['supplier_name']
                                        sell_open_qty = latest_sell_entry['paper_open_quantity']
                                        supplier_name = supplier_name.iloc[0]
                                        sell_open_qty = sell_open_qty.iloc[0]
                                        with col2:
                                            st.info(f"🏢 **Supplier:** {supplier_name}")
                                        with col3:
                                            st.info(f"📊 **Sell Position Open Qty:** {sell_open_qty:.1f} MT")

                                    # Quantity to settle (auto-calculate minimum between positions)
                                    max_hedge_qty = min(paper_open_qty, selected_sell_qty)
                                    # Ensure min_value is valid (must be <= max_value)
                                    min_hedge_qty = min(0.1, max_hedge_qty) if max_hedge_qty > 0 else 0.0
                                    col1,col2 = st.columns([1,2])
                                    with col1:
                                        hedge_quantity = st.number_input(
                                            "Quantity to Settle (MT)",
                                            min_value=float(min_hedge_qty),
                                            max_value=float(max_hedge_qty),
                                            value=float(max_hedge_qty),
                                            step=0.1,
                                            key=f"hedge_qty_{position_id}"
                                        )
                                    # Show what will happen
                                    remaining_buy = float(paper_open_qty) - float(hedge_quantity)
                                    remaining_sell = float(selected_sell_qty) - float(hedge_quantity)
                                    with col2:
                                        st.info(f"📈 **After Settlement:**")
                                        col21, col22 = st.columns(2)
                                        with col21:
                                            st.write(f"• This buy position: {remaining_buy:.1f} MT remaining")
                                        with col22:
                                            st.write(f"• Selected sell position: {remaining_sell:.1f} MT remaining")

                                    # Show settlement impact
                                    if remaining_buy == 0 and remaining_sell == 0:
                                        st.success("🎯 **Perfect Settlement**: Both positions will be completely closed!")
                                    elif remaining_buy == 0:
                                        st.info("✅ **Buy Position Closed**: This buy position will be fully settled")
                                    elif remaining_sell == 0:
                                        st.info("✅ **Sell Position Closed**: The selected sell position will be fully settled")

                                    # Only show hedging form if there's a valid quantity to settle
                                    if max_hedge_qty > 0:
                                        # Create hedging form
                                        with st.form(f"hedge_buy_form_{position_id}"):
                                            if st.form_submit_button("💰 Settle Position"):
                                                # Additional validation: check if quantity exceeds available
                                                if hedge_quantity > paper_open_qty:
                                                    st.error(f"❌ Cannot settle {hedge_quantity:.1f} MT. Only {paper_open_qty:.1f} MT available in this position.")
                                                    return
                                                
                                                try:
                                                    # Settle the positions using the new settlement method
                                                    success = {
                                                        'buy_entry_id': row_df['entry_id'].iloc[0],
                                                        'sell_entry_id': latest_sell_entry['entry_id'].iloc[0],
                                                        'buy_position_id': row_df['position_id'].iloc[0],
                                                        'sell_position_id': latest_sell_entry['position_id'].iloc[0],
                                                        'hedge_quantity': hedge_quantity,
                                                        'product': row_df['product'].iloc[0],
                                                        'buy_price': row_df['price_rate'].iloc[0],
                                                        'sell_price': latest_sell_entry['price_rate'].iloc[0],
                                                        'buy_exchange': row_df['exchange'].iloc[0],
                                                        'sell_exchange': latest_sell_entry['exchange'].iloc[0],
                                                        'settlement_date': datetime.datetime.now().isoformat(),
                                                    }
                                                    if success:
                                                        st.success(f"Position settled: {hedge_quantity:.1f} MT")
                                                        # update postgres 
                                                        try:
                                                            utils.update_table_buy_completly(remaining_buy,row_df['position_id'].iloc[0])
                                                            utils.update_table_sell_completly(remaining_sell,latest_sell_entry['position_id'].iloc[0]) 
                                                        except Exception as e:
                                                            st.error(f"❌ Error updating database: {str(e)}")

                                                        final_settle_df = pd.DataFrame([success]) 
                                                        utils.create_settlement_entry_push(final_settle_df)
                                                        st.rerun()
                                                    else:
                                                        st.error("Failed to settle positions. Please try again.")
                                                except Exception as e:
                                                    st.error(f"❌ Error settling positions: {str(e)}")
                                    
                                    # Settlement results now shown in global section above
                                else:
                                    st.error("Selected position not found")
                                    return
                            else:
                                st.error("Please select a position")
                                return
                    else:
                        st.info(f"📝 No available {commodity} Paper Sell positions to hedge with. Create a {commodity} Paper Sell entry first.")
        
    # =============================================================================
    # PAPER SELL POSITIONS SECTION
    # =============================================================================
    if sell_positions:
        commodity_text = f" ({selected_commodity})" if selected_commodity != "All Commodities" else ""
        st.subheader(f"📉 Paper Sell Positions{commodity_text}")
        
        for position_id, entry_id in sell_positions:
            # Find the row by entry_id
            row_df = paper_sell_df[(paper_sell_df['status'] == 'OPEN') & (paper_sell_df['entry_id'] == entry_id) & (paper_sell_df['position_id'] == position_id)]
            if not row_df.empty:
                row = row_df.iloc[0]

                # Fetch details safely
                commodity = row.get("product", "N/A")
                paper_open_qty = row.get("paper_open_quantity", 0)
                type_ = 'BUY'
                quantity = row.get("total_quantity", 0)
                exchange = row.get("exchange", "N/A")
                rate = row.get("price_rate", 0)
                timestamp = row.get("timestamp", pd.Timestamp.now())
                if pd.notnull(timestamp):
                    timestamp = pd.to_datetime(timestamp).strftime('%d %b %Y')
                transaction_date = row.get("transaction_date", "N/A")
                due_date = row.get("expiry_date", "N/A")
                customer_name = row.get("customer_name", "")
                customer_group_name = row.get("customer_group_name", "")
                num_lots = row.get("num_lots", None)
                lot_size = row.get("lot_size", None)
                supplier_name = row.get("supplier_name", "")
                supplier_group_name = row.get("supplier_group_name", "")

                with st.expander(f"🔴 Sell Position {position_id} - {commodity} ({paper_open_qty:.1f} MT open)", expanded=True):
                    col1, col2, col3,col4,col5,col6 = st.columns([1,1,1,1,1,3])
                    
                    with col1:
                        st.metric("Latest Action", "SELL")
                    with col2:
                        st.metric("Open Quantity", f"{paper_open_qty:.1f} MT")
                    with col3:
                       st.metric("Total Quantity", f"{quantity:.1f} MT")
                    with col4:
                        st.metric("Exchange", exchange)
                    with col5:
                        st.metric("Rate", f"{rate}")
                    with col6:
                        st.metric("Supplier", supplier_group_name) 
                    
                    # =============================================================================
                    # HEDGING FUNCTIONALITY FOR SELL POSITIONS
                    # =============================================================================
                    st.subheader("💰 Settle This Sell Position")
                    # Get all available BUY positions for hedging
                    # Filter buy positions by commodity to match the current sell position
                    matching_buy_positions = paper_buy_df[(paper_buy_df['product'] == commodity) & (paper_buy_df['status'] == 'OPEN')][['position_id', 'entry_id']]
                    
                    if not matching_buy_positions.empty:
                        # Use simple string-based selection to avoid type issues
                        if not matching_buy_positions.empty:
                            # Create simple string options that include the position ID
                            buy_options = [str(position_id) for position_id in matching_buy_positions['position_id'].tolist()]
                            col1,col2,col3 = st.columns([1,1.5,1.5])
                            with col1:
                                selected_buy_string = st.selectbox(
                                    "Select Paper Sell Position to Settle With", 
                                    options=buy_options,  # Use position IDs as strings
                                    key=f"sell_select_{position_id}"
                                )
                            # Parse the selected position ID and get current quantities
                            if selected_buy_string:
                                selected_buy_id = selected_buy_string
                                # Find the corresponding position object
                                selected_buy_position  = matching_buy_positions[matching_buy_positions['position_id'] == selected_buy_string]['entry_id']
                                if not selected_buy_position.empty:
                                    # Get current quantities dynamically
                                    # selected_buy_qty = selected_buy_position.get_open_quantity()
                                    # Auto-populate customer name from the selected buy position
                                    selected_buy_position_value = selected_buy_position.iloc[0]
                                    latest_buy_entry = paper_buy_df[paper_buy_df['entry_id'] == str(selected_buy_position_value)]
                                    selected_buy_qty = latest_buy_entry['paper_open_quantity'].iloc[0]
                                    if not latest_buy_entry.empty:
                                        customer_name = latest_buy_entry['customer_name']
                                        buy_open_qty = latest_buy_entry['paper_open_quantity']
                                        customer_name = customer_name.iloc[0]
                                        buy_open_qty = buy_open_qty.iloc[0]
                                        with col2:
                                            st.info(f"👤 **Customer:** {customer_name}")
                                        with col3:
                                            st.info(f"📊 **Buy Position Open Qty:** {buy_open_qty:.1f} MT")

                                    # Quantity to settle (auto-calculate minimum between positions)
                                    max_hedge_qty = min(paper_open_qty, selected_buy_qty)
                                    # Ensure min_value is valid (must be <= max_value)
                                    min_hedge_qty = min(0.1, max_hedge_qty) if max_hedge_qty > 0 else 0.0
                                    col1,col2 = st.columns([1,2])
                                    with col1:
                                        hedge_quantity = st.number_input(
                                            "Quantity to Settle (MT)",
                                            min_value=float(min_hedge_qty),
                                            max_value=float(max_hedge_qty),
                                            value=float(max_hedge_qty),
                                            step=0.1,
                                            key=f"hedge_qty_{position_id}"
                                        )
                                    
                                    # Show what will happen
                                    remaining_sell = float(paper_open_qty) - float(hedge_quantity)
                                    remaining_buy = float(selected_buy_qty) - float(hedge_quantity)
                                    with col2:
                                        st.info(f"📈 **After Settlement:**")
                                        col21, col22 = st.columns(2)
                                        with col21:
                                            st.write(f"• This sell position: {remaining_sell:.1f} MT remaining")
                                        with col22:
                                            st.write(f"• Selected buy position: {remaining_buy:.1f} MT remaining")

                                    # Show settlement impact
                                    if remaining_sell == 0 and remaining_buy == 0:
                                        st.success("🎯 **Perfect Settlement**: Both positions will be completely closed!")
                                    elif remaining_sell == 0:
                                        st.info("✅ **Sell Position Closed**: This sell position will be fully settled")
                                    elif remaining_buy == 0:
                                        st.info("✅ **Buy Position Closed**: The selected buy position will be fully settled")
                                    
                                    # Only show hedging form if there's a valid quantity to settle
                                    if max_hedge_qty > 0:
                                        # Create hedging form
                                        with st.form(f"hedge_sell_form_{position_id}"):
                                            if st.form_submit_button("💰 Settle Position"):
                                                # Additional validation: check if quantity exceeds available
                                                if hedge_quantity > paper_open_qty:
                                                    st.error(f"❌ Cannot settle {hedge_quantity:.1f} MT. Only {paper_open_qty:.1f} MT available in this position.")
                                                    return
                                                
                                                try:
                                                    # Settle the positions using the new settlement method
                                                    success = {
                                                        'buy_entry_id': latest_buy_entry['entry_id'].iloc[0], 
                                                        'sell_entry_id': row_df['entry_id'].iloc[0],
                                                        'buy_position_id': latest_buy_entry['position_id'].iloc[0],
                                                        'sell_position_id': row_df['position_id'].iloc[0],
                                                        'hedge_quantity': hedge_quantity,
                                                        'product': row_df['product'].iloc[0],
                                                        'buy_price': latest_buy_entry['price_rate'].iloc[0],
                                                        'sell_price': row_df['price_rate'].iloc[0],
                                                        'buy_exchange': latest_buy_entry['exchange'].iloc[0],
                                                        'sell_exchange': row_df['exchange'].iloc[0],
                                                        'settlement_date': datetime.datetime.now().isoformat(),
                                                    }
                                                    if success:
                                                        st.success(f"Position settled: {hedge_quantity:.1f} MT")
                                                        # update postgres 
                                                        try:
                                                            utils.update_table_sell_completly(remaining_sell,row_df['position_id'].iloc[0])
                                                            utils.update_table_buy_completly(remaining_buy,latest_buy_entry['position_id'].iloc[0]) 
                                                        except Exception as e:
                                                            st.error(f"❌ Error updating database: {str(e)}")

                                                        final_settle_df = pd.DataFrame([success]) 
                                                        utils.create_settlement_entry_push(final_settle_df)
                                                        st.rerun()
                                                    else:
                                                        st.error("Failed to settle positions. Please try again.")
                                                except Exception as e:
                                                    st.error(f"❌ Error settling positions: {str(e)}")
                                    
                                    # Settlement results now shown in global section above
                                else:
                                    st.error("Selected position not found")
                                    return
                            else:
                                st.error("Please select a position")
                                return
                        else:
                            st.info(f"📝 No available {commodity} Paper Buy positions to hedge with. Create a {commodity} Paper Buy entry first.")
                    else:
                        st.info(f"📝 No available {commodity} Paper Buy positions to hedge with. Create a {commodity} Paper Buy entry first.")
                    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    if not buy_positions and not sell_positions and not closed_positions:
        if selected_commodity != "All Commodities":
            st.info(f"No {selected_commodity} positions found. Create some {selected_commodity} positions using the Create Paper Buy/Sell tabs first.")
        else:
            st.info("No positions found. Create some positions using the Create Paper Buy/Sell tabs first.")

    # =============================================================================
    # SETTLEMENT FLAG CLEANUP
    # =============================================================================
    # After displaying all positions, clear settlement flags to allow normal navigation
    # Note: We don't clear the flags here anymore since we do it right after displaying the results

def render_position_dashboard(q1,q2,q3):
    """Render the position dashboard sub-tab"""
    st.subheader("Position Dashboard")
    
    sell_df = utils.get_table_data(q1)
    buy_df = utils.get_table_data(q2)
    settle_df = utils.get_table_data(q3)

    product_list = list(set(buy_df['product'].unique().tolist() + sell_df['product'].unique().tolist()))
    product_list.append("All Products")
    product_list.sort()

    col1, col2, col3 = st.columns([2,4,2])
    with col1:
        # Buy sell postion filters
        buy_sell_filter = st.radio(
            "### **Select position type:**",
            ["All", "Buy", "Sell"],
            horizontal=True
        )
    with col2:
        # Product filter
        product = st.radio(
            "### **Select product:**",
            product_list,
            horizontal=True,)
    with col3:
        # expiry data
        expiry_data = st.radio(
            "expiry",
            ["All", "Expiring in 7 or less Days"],
            horizontal=True
            # label_visibility="collapsed"
        )
    
   

    st.markdown("---")
    if buy_sell_filter == "All":
        sell_df = sell_df.drop(['supplier_name','supplier_group_name'], axis=1)
        sell_df['type'] = 'Sell'
        buy_df = buy_df.drop(['customer_name','customer_group_name'], axis=1)
        buy_df['type'] = 'Buy'
        combined_df = pd.concat([sell_df, buy_df], ignore_index=True)
    elif buy_sell_filter == "Buy":
        combined_df = buy_df.drop(['customer_name','customer_group_name'], axis=1)
        combined_df['type'] = 'Buy'
    elif buy_sell_filter == "Sell":
        combined_df = sell_df.drop(['supplier_name','supplier_group_name'], axis=1)
        combined_df['type'] = 'Sell'

    if product != "All Products":
        combined_df = combined_df[combined_df['product'] == product]
        try:
            settle_df = settle_df[settle_df['product'] == product]
        except:
            pass
    if expiry_data == "Expiring in 7 or less Days":
        combined_df = combined_df[pd.to_datetime(combined_df['expiry_date']) <= (pd.Timestamp.now() + pd.Timedelta(days=7))]
    
    # Summary metrics
    total_positions = len(combined_df)
    open_positions = len(combined_df[combined_df['status'] == 'OPEN'])
    closed_positions = total_positions - open_positions

    # open data
    open_df = combined_df[combined_df['status'] == 'OPEN']
    buy_open_df = open_df[open_df['type'] == 'Buy']
    sell_open_df = open_df[open_df['type'] == 'Sell']
    buy_open_no_of_postions = len(buy_open_df)
    sell_no_of_postions = len(sell_open_df)
    buy_open_quantity = buy_open_df['paper_open_quantity'].sum()
    sell_open_quantity = sell_open_df['paper_open_quantity'].sum()
    buy_value = (buy_open_df['paper_open_quantity'] * buy_open_df['price_rate']).sum()
    sell_value = (sell_open_df['paper_open_quantity'] * sell_open_df['price_rate']).sum()

    open_data_buy_df = {
        'Type': ['Buy'],
        'Total Postions': [buy_open_no_of_postions],
        'Open Quantity': [buy_open_quantity],
        'Value': [buy_value]
    }
    open_data_sell_df = {
        'Type': ['Sell'],
        'Total Postions': [sell_no_of_postions],
        'Open Quantity': [sell_open_quantity],
        'Value': [sell_value]
    }
    if buy_sell_filter == 'All':
        open_data_df = pd.concat([pd.DataFrame(open_data_buy_df), pd.DataFrame(open_data_sell_df)], ignore_index=True)
    elif buy_sell_filter == 'Buy':
        open_data_df = pd.DataFrame(open_data_buy_df)
        combined_df = combined_df[combined_df['type'] == 'Buy']
    elif buy_sell_filter == 'Sell':
        open_data_df = pd.DataFrame(open_data_sell_df)
        combined_df = combined_df[combined_df['type'] == 'Sell']
    
    st.subheader("Open Positions")
    st.table(open_data_df)
    st.subheader("Closed Positions")

    # closed data 
    closed_df = combined_df[combined_df['status'].isin(['Closed', 'CLOSED'])]
    buy_closed_df = closed_df[closed_df['type'] == 'Buy']
    sell_closed_df = closed_df[closed_df['type'] == 'Sell']
    buy_closed_no_of_postions = len(buy_closed_df)
    sell_closed_no_of_postions = len(sell_closed_df)
    buy_closed_quantity = buy_closed_df['total_quantity'].sum()
    sell_closed_quantity = sell_closed_df['total_quantity'].sum()
    buy_closed_value = (buy_closed_df['total_quantity'] * buy_closed_df['price_rate']).sum()
    sell_closed_value = (sell_closed_df['total_quantity'] * sell_closed_df['price_rate']).sum()

    closed_data_buy_df = {
        'Type': ['Buy'],
        'Total Postions': [buy_closed_no_of_postions],
        'Open Quantity': [buy_closed_quantity],
        'Value': [buy_closed_value]
    }
    closed_data_sell_df = {
        'Type': ['Sell'],
        'Total Postions': [sell_closed_no_of_postions],
        'Open Quantity': [sell_closed_quantity],
        'Value': [sell_closed_value]
    }

    if buy_sell_filter == 'All':
        closed_data_df = pd.concat([pd.DataFrame(closed_data_buy_df), pd.DataFrame(closed_data_sell_df)], ignore_index=True)
    elif buy_sell_filter == 'Buy':
        closed_data_df = pd.DataFrame(closed_data_buy_df)
    elif buy_sell_filter == 'Sell':
        closed_data_df = pd.DataFrame(closed_data_sell_df)

    st.table(closed_data_df)
    st.markdown("---")
    if not settle_df.empty:
        st.header("Paper Positions Settled")
        cols = ['sell_price_rate', 'buy_price_rate', 'hedge_quantity']
        settle_df_updated = settle_df[['buy_position_id', 'sell_position_id', 'hedge_quantity','product','buy_exchange','buy_price_rate','sell_price_rate','settlement_date']]
        settle_df_updated[cols] = settle_df_updated[cols].apply(pd.to_numeric, errors='coerce')
        settle_df_updated['profit_value'] = (
            settle_df_updated['hedge_quantity'] *
            (settle_df_updated['sell_price_rate'] - settle_df_updated['buy_price_rate'])
            ).round(2)
        settle_df_updated['profit_percentage'] = ((
            (settle_df_updated['sell_price_rate'] - settle_df_updated['buy_price_rate']) * 100
            ) / settle_df_updated['buy_price_rate']).round(2)

        st.table(settle_df_updated)
        st.markdown("---")

    graph_df = combined_df[['position_id','product','total_quantity','price_rate','type']]
    graph1_df = graph_df[['position_id','product','type']]
    count_df = graph1_df.groupby(['product', 'type'])['position_id'].count().reset_index()
    count_df.rename(columns={'position_id': 'position_count'}, inplace=True)

    chart = (
        alt.Chart(count_df)
        .mark_bar()
        .encode(
            x='product:N',
            y='position_count:Q',
            color='type:N',
            tooltip=['product', 'type', 'position_count']
        )
        .properties(
            title='Count of Positions by Product and Type',
            width=700
        )
    )
    
    col2,col1  = st.columns([1,3])
    with col1:
        st.altair_chart(chart)
    with col2:
        if not settle_df.empty:
           graph2_num = (settle_df_updated['hedge_quantity']*(settle_df_updated['sell_price_rate'] - settle_df_updated['buy_price_rate'])).sum()
           graph3_num = (settle_df_updated['hedge_quantity']*( settle_df_updated['buy_price_rate'])).sum()
           graph4_num = (graph2_num*100)/graph3_num
           st.metric(label="Overall Profit ", value=f"{graph2_num:.2f}")
           st.metric(label="Overall Profit Percentage", value=f"{graph4_num:.2f}%")
        #    st.table(settle_df_updated) 

    st.markdown("---")

def render_rollover_positions(q1,q2):
    col1,col2 = st.columns([8,1])
    with col1:
        st.subheader("**Roll-On Positions**")
    with col2:
        # Placeholder for future implementation
        st.button("⚠️", help="Roll-On functionality is under development and will be available soon.")
    
    base_paper_buy_df = utils.get_table_data(q2)
    base_paper_sell_df = utils.get_table_data(q1)

    # take what option to roll over
    opt = st.selectbox("Select Positions to Roll Over", options=["Paper Buy", "Paper Sell"])

    if opt == 'Paper Buy':
        paper_buy_df = base_paper_buy_df[base_paper_buy_df['status'] == 'OPEN']
        paper_buy_df['filter_flag'] = 'Position ' + paper_buy_df['position_id']+' of Customer '+paper_buy_df['customer_name']+' for item '+paper_buy_df['product'] + ' with Open Qty '+paper_buy_df['paper_open_quantity'].astype(str)+' MT'
        pb_filter_list = paper_buy_df['filter_flag'].unique().tolist()
        selected_pb_filter = st.selectbox("Select Paper Buy Position to Roll Over", options=pb_filter_list)
        if selected_pb_filter:
            filtered_paper_buy = paper_buy_df[paper_buy_df['filter_flag'] == selected_pb_filter]
            max_roll_qty = filtered_paper_buy['paper_open_quantity'].iloc[0]
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                roll_qty = st.number_input("Enter Quantity to Roll Over (MT)", min_value=0.1, step=0.1,max_value=float(max_roll_qty))
            with col2:
                updated_price = st.number_input("Enter New Price Rate", min_value=0.0, step=0.01,value=float(filtered_paper_buy['price_rate'].iloc[0]))
            with col3:
                new_expiry_date = st.date_input("Select New Expiry Date", min_value=filtered_paper_buy['expiry_date'].iloc[0] + datetime.timedelta(days=30))
            if st.button("🔄 Roll Over Paper Buy Position"):
                if roll_qty <= 0 or roll_qty > max_roll_qty:
                    st.error(f"❌ Invalid roll over quantity. Must be between 0.1 and {max_roll_qty:.1f} MT.")
                else:
                    try:
                        # Update the existing position's open quantity
                        new_open_qty = float(max_roll_qty) - float(roll_qty)
                        # utils.update_table_buy_completly(new_open_qty,filtered_paper_buy['position_id'].iloc[0])
                        
                        # Create a new position with the rolled over details
                        timestamp = int(time.time() * 1000)  # milliseconds since epoch
                        rand_part = random.randint(1000, 9999)
                        unique_number = f"{timestamp}{rand_part}"[-10:]  # keep last 10 digits for consistency

                        # Final position ID
                        position_id = f"HEDGE_{unique_number}"
                        existing_sell_entity_ids = base_paper_sell_df['entry_id'].unique().tolist()
                        existing_buy_entity_ids = base_paper_buy_df['entry_id'].unique().tolist()
                        sell_entry_id = generate_entity_id(existing_sell_entity_ids)
                        buy_entry_id = generate_entity_id(existing_buy_entity_ids)
                        filtered_paper_buy['paper_open_quantity'].iloc[0] = new_open_qty
                        filtered_paper_buy['roll_over_quantity'].iloc[0] = float(filtered_paper_buy['roll_over_quantity'].iloc[0]) + float(roll_qty)
                      
                        new_entry_buy = {
                            'entry_id': buy_entry_id,
                            'position_id': position_id,
                            'product': filtered_paper_buy['product'].iloc[0],
                            'status': 'OPEN',
                            'artefact_status': 'OPEN',
                            'paper_open_quantity': roll_qty,
                            'total_quantity': roll_qty,
                            'invoice_open_quantity': roll_qty,
                            'price_rate': updated_price,
                            'exchange': filtered_paper_buy['exchange'].iloc[0],
                            'transaction_date': datetime.datetime.now().date().isoformat(),
                            'expiry_date': new_expiry_date,
                            'customer_name': filtered_paper_buy['customer_name'].iloc[0],
                            'customer_group_name': filtered_paper_buy['customer_group_name'].iloc[0],
                            'num_lots': filtered_paper_buy['num_lots'].iloc[0],
                            'lot_size': filtered_paper_buy['lot_size'].iloc[0],
                            'roll_over_quantity': 0.0,
                            'currency': filtered_paper_buy['currency'].iloc[0],
                            'conversion_rate': filtered_paper_buy['conversion_rate'].iloc[0],
                        }
                       
                        new_entry_sell = {
                            'entry_id': sell_entry_id,
                            'position_id': position_id,
                            'product': filtered_paper_buy['product'].iloc[0],
                            'status': 'CLOSED',
                            'artefact_status': 'OPEN',
                            'paper_open_quantity': 0.0,
                            'total_quantity': roll_qty,
                            'bill_open_quantity': roll_qty,
                            'price_rate': updated_price,
                            'exchange': filtered_paper_buy['exchange'].iloc[0],
                            'transaction_date': datetime.datetime.now().date().isoformat(),
                            'expiry_date': new_expiry_date,
                            'num_lots': filtered_paper_buy['num_lots'].iloc[0],
                            'lot_size': filtered_paper_buy['lot_size'].iloc[0],
                            'supplier_name': filtered_paper_buy['customer_name'].iloc[0],
                            'supplier_group_name': filtered_paper_buy['customer_group_name'].iloc[0],
                            'roll_over_quantity': roll_qty,
                            'currency': filtered_paper_buy['currency'].iloc[0],
                            'conversion_rate': filtered_paper_buy['conversion_rate'].iloc[0],
                        }
                       
                        comb_entry = {
                            'old_entry_id': filtered_paper_buy['entry_id'].iloc[0],
                            'old_position_id': filtered_paper_buy['position_id'].iloc[0],
                            'new_entry_id': position_id,
                            'buy_position_id': buy_entry_id,
                            'sell_position_id': sell_entry_id,
                            'roll_over_quantity': roll_qty,
                            'product': filtered_paper_buy['product'].iloc[0],
                            'price_rate': updated_price,
                            'type': 'Buy',
                        }
                       
                        new_entry_buy_df = pd.DataFrame([new_entry_buy])
                        new_entry_sell_df = pd.DataFrame([new_entry_sell])
                        comb_entry_df = pd.DataFrame([comb_entry])
                        utils.create_paper_roll_over_tagging(comb_entry_df)
                        utils.update_table_buy_roll_over_completly(new_open_qty,filtered_paper_buy['roll_over_quantity'].iloc[0],filtered_paper_buy['position_id'].iloc[0])
                        utils.create_paper_buy_entry_push(new_entry_buy_df)
                        utils.create_paper_sell_entry_push(new_entry_sell_df)
                        st.rerun()
                        st.success(f"✅ Successfully rolled over {roll_qty:.1f} MT to new Paper Buy position.")
                    except Exception as e:  
                        st.error(f"❌ Error during roll-over: {str(e)}")
    elif opt == 'Paper Sell':
        paper_sell_df = base_paper_sell_df[base_paper_sell_df['status'] == 'OPEN']
        paper_sell_df['filter_flag'] = 'Position ' + paper_sell_df['position_id']+' of Supplier '+paper_sell_df['supplier_name']+' for item '+paper_sell_df['product'] + ' with Open Qty '+paper_sell_df['paper_open_quantity'].astype(str)+' MT'
        ps_filter_list = paper_sell_df['filter_flag'].unique().tolist()
        selected_ps_filter = st.selectbox("Select Paper Sell Position to Roll Over", options=ps_filter_list)
        if selected_ps_filter:
            filtered_paper_sell = paper_sell_df[paper_sell_df['filter_flag'] == selected_ps_filter]
            max_roll_qty = filtered_paper_sell['paper_open_quantity'].iloc[0]
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                roll_qty = st.number_input("Enter Quantity to Roll Over (MT)", min_value=0.1, step=0.1,max_value=float(max_roll_qty))
            with col2:
                updated_price = st.number_input("Enter New Price Rate", min_value=0.0, step=0.01,value=float(filtered_paper_sell['price_rate'].iloc[0]))
            with col3:
                new_expiry_date = st.date_input("Select New Expiry Date", min_value=filtered_paper_sell['expiry_date'].iloc[0] + datetime.timedelta(days=30))
            if st.button("🔄 Roll Over Paper Sell Position"):
                if roll_qty <= 0 or roll_qty > max_roll_qty:
                    st.error(f"❌ Invalid roll over quantity. Must be between 0.1 and {max_roll_qty:.1f} MT.")
                else:
                    try:
                        new_open_qty = float(max_roll_qty) - float(roll_qty)
                        timestamp = int(time.time() * 1000)  # milliseconds since epoch
                        rand_part = random.randint(1000, 9999)
                        unique_number = f"{timestamp}{rand_part}"[-10:]  # keep last 10 digits for consistency  

                        #final position ID
                        position_id = f"HEDGE_{unique_number}"
                        existing_sell_entity_ids = base_paper_sell_df['entry_id'].unique().tolist()
                        existing_buy_entity_ids = base_paper_buy_df['entry_id'].unique().tolist()
                        sell_entry_id = generate_entity_id(existing_sell_entity_ids)
                        buy_entry_id = generate_entity_id(existing_buy_entity_ids)
                        filtered_paper_sell['paper_open_quantity'].iloc[0] = new_open_qty
                        filtered_paper_sell['roll_over_quantity'].iloc[0] = float(filtered_paper_sell['roll_over_quantity'].iloc[0]) + float(roll_qty)
                        
                        new_entry_sell = {
                            'entry_id': sell_entry_id,
                            'position_id': position_id,
                            'product': filtered_paper_sell['product'].iloc[0],
                            'status': 'OPEN',
                            'artefact_status': 'OPEN',
                            'paper_open_quantity': roll_qty,
                            'total_quantity': roll_qty,
                            'bill_open_quantity': roll_qty,
                            'price_rate': updated_price,
                            'exchange': filtered_paper_sell['exchange'].iloc[0],
                            'transaction_date': datetime.datetime.now().date().isoformat(),
                            'expiry_date': new_expiry_date,
                            'supplier_name': filtered_paper_sell['supplier_name'].iloc[0],
                            'supplier_group_name': filtered_paper_sell['supplier_group_name'].iloc[0],
                            'num_lots': filtered_paper_sell['num_lots'].iloc[0],
                            'lot_size': filtered_paper_sell['lot_size'].iloc[0],
                            'roll_over_quantity': 0.0,
                            'currency': filtered_paper_sell['currency'].iloc[0],
                            'conversion_rate': filtered_paper_sell['conversion_rate'].iloc[0],
                        }
                        
                        new_entry_buy = {
                            'entry_id': buy_entry_id,
                            'position_id': position_id,
                            'product': filtered_paper_sell['product'].iloc[0],
                            'status': 'CLOSED',
                            'artefact_status': 'OPEN',
                            'paper_open_quantity': 0.0,
                            'total_quantity': roll_qty,
                            'invoice_open_quantity': roll_qty,
                            'price_rate': updated_price,
                            'exchange': filtered_paper_sell['exchange'].iloc[0],
                            'transaction_date': datetime.datetime.now().date().isoformat(),
                            'expiry_date': new_expiry_date,
                            'customer_name': filtered_paper_sell['supplier_name'].iloc[0],
                            'customer_group_name': filtered_paper_sell['supplier_group_name'].iloc[0],
                            'num_lots': filtered_paper_sell['num_lots'].iloc[0],
                            'lot_size': filtered_paper_sell['lot_size'].iloc[0],
                            'roll_over_quantity': roll_qty,
                            'currency': filtered_paper_sell['currency'].iloc[0],
                            'conversion_rate': filtered_paper_sell['conversion_rate'].iloc[0],
                        }
                        
                        comb_entry = {
                            'old_entry_id': filtered_paper_sell['entry_id'].iloc[0],
                            'old_position_id': filtered_paper_sell['position_id'].iloc[0],
                            'new_entry_id': position_id,
                            'buy_position_id': buy_entry_id,
                            'sell_position_id': sell_entry_id,
                            'roll_over_quantity': roll_qty,
                            'product': filtered_paper_sell['product'].iloc[0],
                            'price_rate': updated_price,
                            'type': 'Sell',
                        }
                        
                        new_entry_sell_df = pd.DataFrame([new_entry_sell])
                        new_entry_buy_df = pd.DataFrame([new_entry_buy])
                        comb_entry_df = pd.DataFrame([comb_entry])
                        utils.create_paper_roll_over_tagging(comb_entry_df)
                        utils.update_table_sell_roll_over_completly(new_open_qty,filtered_paper_sell['roll_over_quantity'].iloc[0],filtered_paper_sell['position_id'].iloc[0])  
                        utils.create_paper_sell_entry_push(new_entry_sell_df)
                        utils.create_paper_buy_entry_push(new_entry_buy_df)
                        st.rerun()
                        st.success(f"✅ Successfully rolled over {roll_qty:.1f} MT to new Paper Sell position.")

                    except Exception as e:  
                        st.error(f"❌ Error during roll-over: {str(e)}")


    else:
        return


def render_profit_loss_tab():
    """Render the profit and loss analysis tab"""
    st.subheader("Profit & Loss Analysis")
    st.info("This section provides an analysis of profit and loss from settled hedging positions.")
    # Placeholder for future implementation
    st.warning("⚠️ P&L Analysis functionality is under development and will be available soon.")
 
# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Set wide layout
st.set_page_config(page_title="Hedging Module", layout="wide")

# ====== Custom CSS ======
st.markdown("""
    <style>
    /* Center the main title */
    .centered-title {
        text-align: center;
        font-size: 3em !important;
        font-weight: 800 !important;
        color: white !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Adjust Streamlit tabs */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-evenly;
        border-bottom: 1 !important;   /* remove Streamlit’s default bottom line */
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        flex: 1;
        text-align: center;
        font-size: 1.2em;
        font-weight: 500;
        color: white !important;
        background-color: transparent;
        border: 1px solid #808080 !important;  /* ensures no hidden border remains */
    }

    /* Active tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff4b4b !important;
        border-bottom: 3px solid #ff4b4b !important;
        font-weight: 700 !important;
    }

    /* Hover effect */
    button[data-baseweb="tab"]:hover {
        color: #ff7676 !important;
    }

    /* Completely hide default bottom border line */
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====== Title ======
st.markdown("<h1 class='centered-title'>🏦 Hedging Module</h1>", unsafe_allow_html=True)

# ====== Functional Tabs ======
inv_tagging, bill_tagging, hedge_management, profit_loss_tab = st.tabs([
    "📄 Invoice Tagging", 
    "📋 Bill Tagging", 
    "📊 Hedging Management",
    "📈 P&L Analysis"
])

# Create sub-tabs for hedging operations
# inv_tagging, bill_tagging, hedge_management, profit_loss_tab = st.tabs(["📄 Invoice Tagging", "📋 Bill Tagging", "📊 Hedging Management","P&L Analysis"])

# Main content area
with inv_tagging:
    render_invoice_tagging_tab(8967,'paper_buy_entry','paper_invoice_tagging')
with bill_tagging:
    render_bill_tagging_tab(8968,'paper_sell_entry','paper_bill_tagging')
with hedge_management:
    render_hedging_tab()
with profit_loss_tab:
    # render_profit_loss_tab()
    render_position_dashboard("paper_sell_entry","paper_buy_entry","hedge_paper_settle_entry")



# # Sidebar info
# st.sidebar.subheader("Quick Stats")
# st.sidebar.metric("Total Positions", len(hedging_module.positions))
# st.sidebar.metric("Open Positions", len(hedging_module.get_open_positions()))
# st.sidebar.metric("Total Bills", len(hedging_module.bills))
# st.sidebar.metric("Total Invoices", len(hedging_module.invoices))

# Testing functionality
# if st.sidebar.button("🧪 Generate Test Data"):
#     dummy_positions = generate_dummy_paper_entries(hedging_module)
#     st.sidebar.success(f"✅ Test data generated! {len(hedging_module.positions)} positions created")
    # st.rerun()

# Refresh data functionality
# if st.sidebar.button("🔄 Refresh Data"):
#     # Clean up all form-related session state variables
#     form_vars = [
#         'ps_lots_input', 'ps_lot_size_input', 'ps_price_input', 'ps_expiry_input', 'ps_transaction_input',
#         'ps_supplier_input', 'ps_conversion_input', 'ps_form_success', 'ps_form_reset_counter',
#         'pb_lots_input', 'pb_lot_size_input', 'pb_price_input', 'pb_expiry_input', 'pb_transaction_input',
#         'pb_customer_input', 'pb_conversion_input', 'pb_form_success', 'pb_form_reset_counter'
#     ]
    
#     for var in form_vars:
#         if var in st.session_state:
#             del st.session_state[var]
#     print("Droped all data points")
#     utils.drop_data_papaer()
#     st.session_state.hedging_module = HedgingModule()
#     bills, invoices = generate_mock_data()
#     st.session_state.hedging_module.bills = bills
#     st.session_state.hedging_module.invoices = invoices
    
#     st.success("Data refreshed successfully!")

if __name__ == "__main__":
    # The main application logic is now at module level
    # Streamlit will execute the code when the script is run
    pass
