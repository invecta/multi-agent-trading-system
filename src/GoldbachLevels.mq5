//+------------------------------------------------------------------+
//|                                           GoldbachTradingSystem.mq5 |
//|                                  Copyright 2025, Goldbach Trading   |
//|                                             https://www.goldbach.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Goldbach Trading"
#property link      "https://www.goldbach.com"
#property version   "1.00"
#property description "Goldbach Trading System based on Power of Three (PO3) and Goldbach levels"

//--- Input Parameters
input group "=== Goldbach System Parameters ==="
input double   PO3_Multiplier = 3.0;           // Power of Three multiplier
input double   Goldbach_Multiplier = 6.0;      // Goldbach level multiplier
input int      LookBack_Period = 9;            // Look back period (Tesla number 9)
input double   PO3_DealingRange = 100.0;      // Base PO3 dealing range in pips
input double   Goldbach_Levels = 6.0;         // Number of Goldbach levels to calculate

input group "=== Trading Parameters ==="
input double   LotSize = 0.1;                  // Trading lot size
input int      StopLoss_Pips = 50;             // Stop loss in pips
input int      TakeProfit_Pips = 100;          // Take profit in pips
input bool     UseTrailingStop = true;         // Use trailing stop
input int      TrailingStop_Pips = 30;         // Trailing stop distance
input int      MagicNumber = 12345;            // Magic number for trades

input group "=== Risk Management ==="
input double   MaxRiskPercent = 2.0;           // Maximum risk per trade (%)
input int      MaxOpenTrades = 3;              // Maximum open trades
input bool     UseTimeFilter = true;            // Use time-based filtering
input int      StartHour = 8;                  // Trading start hour (GMT)
input int      EndHour = 20;                   // Trading end hour (GMT)

//--- Global Variables
double g_goldbachLevels[];                     // Array to store Goldbach levels
double g_po3Levels[];                          // Array to store PO3 levels
datetime g_lastTradeTime = 0;                  // Last trade time
int g_totalTrades = 0;                         // Total trades counter
double g_balance = 0;                          // Account balance

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== Goldbach Trading System Initialized ===");
   Print("PO3 Multiplier: ", PO3_Multiplier);
   Print("Goldbach Multiplier: ", Goldbach_Multiplier);
   Print("Look Back Period: ", LookBack_Period);
   
   // Initialize arrays
   ArrayResize(g_goldbachLevels, (int)Goldbach_Levels);
   ArrayResize(g_po3Levels, 3); // PO3 has 3 main levels
   
   // Calculate initial levels
   CalculateGoldbachLevels();
   CalculatePO3Levels();
   
   g_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("=== Goldbach Trading System Deinitialized ===");
   Print("Total trades executed: ", g_totalTrades);
}

//+------------------------------------------------------------------+
//| Expert tick function                                           |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if we should trade based on time filter
   if(UseTimeFilter && !IsTradingTime())
      return;
      
   // Update levels
   CalculateGoldbachLevels();
   CalculatePO3Levels();
   
   // Check for trading opportunities
   CheckForTradeSignals();
   
   // Manage open positions
   ManageOpenPositions();
}

//+------------------------------------------------------------------+
//| Calculate Goldbach Levels based on mathematical concept        |
//+------------------------------------------------------------------+
void CalculateGoldbachLevels()
{
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = iATR(_Symbol, PERIOD_CURRENT, 14);
   for(int i = 0; i < (int)Goldbach_Levels; i++)
   {
      // Goldbach formula: Base + (i * 6 * ATR)
      g_goldbachLevels[i] = currentPrice + (i * Goldbach_Multiplier * atr);
   }
   
   // Also calculate levels below current price
   for(int i = 0; i < (int)Goldbach_Levels; i++)
   {
      g_goldbachLevels[i] = currentPrice - (i * Goldbach_Multiplier * atr);
   }
}

//+------------------------------------------------------------------+
//| Calculate Power of Three (PO3) Levels                         |
//+------------------------------------------------------------------+
void CalculatePO3Levels()
{
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double baseRange = PO3_DealingRange * _Point * 10; // Convert pips to price
   
   // PO3 levels: Base, Base*3, Base*9 (Tesla numbers)
   g_po3Levels[0] = currentPrice;                    // Base level
   g_po3Levels[1] = currentPrice + (baseRange * 3);  // PO3 level 1
   g_po3Levels[2] = currentPrice + (baseRange * 9);  // PO3 level 2
}

//+------------------------------------------------------------------+
//| Check for trading signals based on Goldbach system            |
//+------------------------------------------------------------------+
void CheckForTradeSignals()
{
   if(CountOpenTrades() >= MaxOpenTrades)
      return;
      
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Check for buy signals at Goldbach support levels
   for(int i = 0; i < (int)Goldbach_Levels; i++)
   {
      if(MathAbs(currentPrice - g_goldbachLevels[i]) < (10 * _Point)) // Within 10 pips
      {
         if(IsGoldbachSupportLevel(i))
         {
            ExecuteBuyOrder(i);
            break;
         }
      }
   }
   
   // Check for sell signals at Goldbach resistance levels
   for(int i = 0; i < (int)Goldbach_Levels; i++)
   {
      if(MathAbs(currentPrice - g_goldbachLevels[i]) < (10 * _Point)) // Within 10 pips
      {
         if(IsGoldbachResistanceLevel(i))
         {
            ExecuteSellOrder(i);
            break;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if level is a Goldbach support level                    |
//+------------------------------------------------------------------+
bool IsGoldbachSupportLevel(int levelIndex)
{
   // Goldbach support: Price bounces from level with number 6 pattern
   double level = g_goldbachLevels[levelIndex];
   double atr = iATR(_Symbol, PERIOD_CURRENT, 14);
   
   // Check if price has tested this level multiple times (6 pattern)
   int touchCount = CountPriceTouches(level, LookBack_Period);
   
   return (touchCount >= 6); // Goldbach number
}

//+------------------------------------------------------------------+
//| Check if level is a Goldbach resistance level                 |
//+------------------------------------------------------------------+
bool IsGoldbachResistanceLevel(int levelIndex)
{
   // Goldbach resistance: Price rejected from level with number 6 pattern
   double level = g_goldbachLevels[levelIndex];
   
   // Check if price has been rejected from this level multiple times
   int rejectionCount = CountPriceRejections(level, LookBack_Period);
   
   return (rejectionCount >= 6); // Goldbach number
}

//+------------------------------------------------------------------+
//| Count how many times price touched a specific level           |
//+------------------------------------------------------------------+
int CountPriceTouches(double level, int periods)
{
   int count = 0;
   
   for(int i = 1; i <= periods; i++)
   {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      
      // Check if price touched the level
      if((high >= level && low <= level) || 
         MathAbs(high - level) < (5 * _Point) || 
         MathAbs(low - level) < (5 * _Point))
      {
         count++;
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| Count how many times price was rejected from a level          |
//+------------------------------------------------------------------+
int CountPriceRejections(double level, int periods)
{
   int count = 0;
   
   for(int i = 1; i <= periods; i++)
   {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      double close = iClose(_Symbol, PERIOD_CURRENT, i);
      
      // Check for rejection pattern (price approaches level but closes away)
      if((high >= level && close < level) || 
         (low <= level && close > level))
      {
         count++;
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| Validate and adjust stop loss level                           |
//+------------------------------------------------------------------+
double ValidateAndAdjustStopLoss(double entryPrice, int stopLossPips, bool isBuy)
{
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Get broker's minimum stop level distance
   long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDistance = stopsLevel * _Point;
   
   double stopLoss = 0;
   
   if(isBuy)
   {
      // For buy orders: SL below entry price
      stopLoss = entryPrice - (stopLossPips * _Point * 10);
      
      // Ensure SL is below current bid by minimum distance
      if(stopLoss >= (currentBid - minDistance))
      {
         stopLoss = currentBid - minDistance;
         Print("Adjusting buy SL to meet minimum distance: ", stopLoss);
      }
      
      // Additional safety check
      if(stopLoss >= currentBid)
      {
         Print("ERROR: Buy SL ", stopLoss, " is above or equal to current bid ", currentBid);
         return 0; // Invalid level
      }
   }
   else
   {
      // For sell orders: SL above entry price
      stopLoss = entryPrice + (stopLossPips * _Point * 10);
      
      // Ensure SL is above current ask by minimum distance
      if(stopLoss <= (currentAsk + minDistance))
      {
         stopLoss = currentAsk + minDistance;
         Print("Adjusting sell SL to meet minimum distance: ", stopLoss);
      }
      
      // Additional safety check
      if(stopLoss <= currentAsk)
      {
         Print("ERROR: Sell SL ", stopLoss, " is below or equal to current ask ", currentAsk);
         return 0; // Invalid level
      }
   }
   
   return stopLoss;
}

//+------------------------------------------------------------------+
//| Validate and adjust take profit level                         |
//+------------------------------------------------------------------+
double ValidateAndAdjustTakeProfit(double entryPrice, int takeProfitPips, bool isBuy)
{
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Get broker's minimum stop level distance
   long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDistance = stopsLevel * _Point;
   
   double takeProfit = 0;
   
   if(isBuy)
   {
      // For buy orders: TP above entry price
      takeProfit = entryPrice + (takeProfitPips * _Point * 10);
      
      // Ensure TP is above current ask by minimum distance
      if(takeProfit <= (currentAsk + minDistance))
      {
         takeProfit = currentAsk + minDistance;
         Print("Adjusting buy TP to meet minimum distance: ", takeProfit);
      }
      
      // Additional safety check
      if(takeProfit <= currentAsk)
      {
         Print("ERROR: Buy TP ", takeProfit, " is below or equal to current ask ", currentAsk);
         return 0; // Invalid level
      }
   }
   else
   {
      // For sell orders: TP below entry price
      takeProfit = entryPrice - (takeProfitPips * _Point * 10);
      
      // Ensure TP is below current bid by minimum distance
      if(takeProfit >= (currentBid - minDistance))
      {
         takeProfit = currentBid - minDistance;
         Print("Adjusting sell TP to meet minimum distance: ", takeProfit);
      }
      
      // Additional safety check
      if(takeProfit >= currentBid)
      {
         Print("ERROR: Sell TP ", takeProfit, " is above or equal to current bid ", currentBid);
         return 0; // Invalid level
      }
   }
   
   return takeProfit;
}

//+------------------------------------------------------------------+
//| Execute buy order based on Goldbach system                    |
//+------------------------------------------------------------------+
void ExecuteBuyOrder(int levelIndex)
{
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Validate and adjust stop loss and take profit
   double stopLoss = ValidateAndAdjustStopLoss(currentAsk, StopLoss_Pips, true);  // true for buy
   double takeProfit = ValidateAndAdjustTakeProfit(currentAsk, TakeProfit_Pips, true);  // true for buy
   
   // Check if we have valid levels
   if(stopLoss == 0 || takeProfit == 0)
   {
      Print("Invalid stop loss or take profit levels for buy order at level ", levelIndex);
      return;
   }
   
   // Adjust levels based on PO3 dealing range
   if(IsWithinPO3Range(currentAsk))
   {
      // Execute the trade
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = LotSize;
      request.type = ORDER_TYPE_BUY;
      request.price = currentAsk;
      request.sl = stopLoss;
      request.tp = takeProfit;
      request.deviation = 10;
      request.magic = MagicNumber;
      request.comment = "Goldbach Buy Level " + IntegerToString(levelIndex);
      
      Print("Executing buy order - Ask: ", currentAsk, ", SL: ", stopLoss, ", TP: ", takeProfit);
      
      if(OrderSend(request, result))
      {
         Print("Buy order executed at level ", levelIndex, " Price: ", currentAsk);
         g_totalTrades++;
         g_lastTradeTime = TimeCurrent();
      }
      else
      {
         Print("Buy order failed. Error: ", result.retcode, " - ", result.comment);
      }
   }
}

//+------------------------------------------------------------------+
//| Execute sell order based on Goldbach system                   |
//+------------------------------------------------------------------+
void ExecuteSellOrder(int levelIndex)
{
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Validate and adjust stop loss and take profit
   double stopLoss = ValidateAndAdjustStopLoss(currentBid, StopLoss_Pips, false);  // false for sell
   double takeProfit = ValidateAndAdjustTakeProfit(currentBid, TakeProfit_Pips, false);  // false for sell
   
   // Check if we have valid levels
   if(stopLoss == 0 || takeProfit == 0)
   {
      Print("Invalid stop loss or take profit levels for sell order at level ", levelIndex);
      return;
   }
   
   // Adjust levels based on PO3 dealing range
   if(IsWithinPO3Range(currentBid))
   {
      // Execute the trade
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = LotSize;
      request.type = ORDER_TYPE_SELL;
      request.price = currentBid;
      request.sl = stopLoss;
      request.tp = takeProfit;
      request.deviation = 10;
      request.magic = MagicNumber;
      request.comment = "Goldbach Sell Level " + IntegerToString(levelIndex);
      
      Print("Executing sell order - Bid: ", currentBid, ", SL: ", stopLoss, ", TP: ", takeProfit);
      
      if(OrderSend(request, result))
      {
         Print("Sell order executed at level ", levelIndex, " Price: ", currentBid);
         g_totalTrades++;
         g_lastTradeTime = TimeCurrent();
      }
      else
      {
         Print("Sell order failed. Error: ", result.retcode, " - ", result.comment);
      }
   }
}

//+------------------------------------------------------------------+
//| Check if trailing stop modification is beneficial             |
//+------------------------------------------------------------------+
bool IsTrailingStopBeneficial(double currentSL, double newSL, double openPrice, bool isBuy)
{
   // Don't modify if current SL is 0 (no SL set)
   if(currentSL == 0)
      return true;
   
   // Calculate the difference in pips
   double difference = MathAbs(newSL - currentSL) / (_Point * 10);
   
   // Only modify if the difference is significant (at least 5 pips)
   if(difference < 5)
   {
      Print("Trailing stop skipped - difference too small: ", difference, " pips");
      return false;
   }
   
   // For buy positions: ensure new SL is higher (better) than current SL
   if(isBuy)
   {
      if(newSL <= currentSL)
      {
         Print("Trailing stop skipped - new SL ", newSL, " not better than current SL ", currentSL);
         return false;
      }
   }
   // For sell positions: ensure new SL is lower (better) than current SL
   else
   {
      if(newSL >= currentSL)
      {
         Print("Trailing stop skipped - new SL ", newSL, " not better than current SL ", currentSL);
         return false;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if price is within PO3 dealing range                    |
//+------------------------------------------------------------------+
bool IsWithinPO3Range(double price)
{
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double baseRange = PO3_DealingRange * _Point * 10;
   
   // Check if price is within optimal PO3 dealing range
   return (MathAbs(price - currentPrice) <= (baseRange * 3));
}

//+------------------------------------------------------------------+
//| Manage open positions with trailing stop                      |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   if(!UseTrailingStop)
      return;
      
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            double currentSL = PositionGetDouble(POSITION_SL);
            double currentTP = PositionGetDouble(POSITION_TP);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            if(posType == POSITION_TYPE_BUY)
            {
               // Trailing stop for buy position
               double newSL = currentPrice - (TrailingStop_Pips * _Point * 10);
               
               // Additional validation: ensure new SL is not too close to current price
               long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
               double minDistance = stopsLevel * _Point;
               
               // Enhanced validation for buy positions
               if(newSL > currentSL && newSL > openPrice && newSL < (currentPrice - minDistance))
               {
                  // Check if the trailing stop modification is actually beneficial
                  if(IsTrailingStopBeneficial(currentSL, newSL, openPrice, true))
                  {
                     Print("Trailing stop triggered for BUY position ", ticket);
                     Print("Current SL: ", currentSL, ", New SL: ", newSL, ", Current Price: ", currentPrice);
                     ModifyPositionSL(newSL, ticket);
                  }
               }
            }
            else if(posType == POSITION_TYPE_SELL)
            {
               // Trailing stop for sell position
               double newSL = currentPrice + (TrailingStop_Pips * _Point * 10);
               
               // Additional validation: ensure new SL is not too close to current price
               long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
               double minDistance = stopsLevel * _Point;
               
               // Enhanced validation for sell positions
               if((newSL < currentSL || currentSL == 0) && newSL < openPrice && newSL > (currentPrice + minDistance))
               {
                  // Check if the trailing stop modification is actually beneficial
                  if(IsTrailingStopBeneficial(currentSL, newSL, openPrice, false))
                  {
                     Print("Trailing stop triggered for SELL position ", ticket);
                     Print("Current SL: ", currentSL, ", New SL: ", newSL, ", Current Price: ", currentPrice);
                     ModifyPositionSL(newSL, ticket);
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Modify position stop loss                                     |
//+------------------------------------------------------------------+
void ModifyPositionSL(double newSL, ulong positionTicket)
{
   if(PositionSelectByTicket(positionTicket))
   {
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      
      // Check if the new SL is actually different from current SL
      if(MathAbs(newSL - currentSL) < (1 * _Point)) // Less than 1 pip difference
      {
         Print("Skipping SL modification - new SL ", newSL, " is too close to current SL ", currentSL, " for position ", positionTicket);
         return;
      }
      
      // Get broker's minimum stop level distance
      long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
      double minDistance = stopsLevel * _Point;
      
      // Get current bid/ask for more accurate validation
      double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      Print("=== SL Modification Request ===");
      Print("Position: ", positionTicket, ", Type: ", (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"));
      Print("Current SL: ", currentSL, ", Requested SL: ", newSL);
      Print("Current Price: ", currentPrice, ", Bid: ", currentBid, ", Ask: ", currentAsk);
      Print("Stops Level: ", stopsLevel, ", Min Distance: ", minDistance);
      
      // Validate and adjust stop loss based on position type
      bool isValidSL = false;
      double adjustedSL = newSL;
      
      if(posType == POSITION_TYPE_BUY)
      {
         // For buy positions: SL must be below current bid by minimum distance
         // Also ensure it's not above the current ask (which would be invalid)
         if(newSL < (currentBid - minDistance) && newSL < currentAsk)
         {
            isValidSL = true;
            Print("Buy SL validation passed - no adjustment needed");
         }
         else
         {
            // Adjust SL to meet minimum distance requirement and be below ask
            adjustedSL = MathMin(currentBid - minDistance, currentAsk - minDistance);
            Print("Adjusting buy SL from ", newSL, " to ", adjustedSL, " (min distance: ", minDistance, " pips)");
            Print("Current Bid: ", currentBid, ", Current Ask: ", currentAsk);
         }
      }
      else if(posType == POSITION_TYPE_SELL)
      {
         // For sell positions: SL must be above current ask by minimum distance
         // Also ensure it's not below the current bid (which would be invalid)
         if(newSL > (currentAsk + minDistance) && newSL > currentBid)
         {
            isValidSL = true;
            Print("Sell SL validation passed - no adjustment needed");
         }
         else
         {
            // Adjust SL to meet minimum distance requirement and be above bid
            adjustedSL = MathMax(currentAsk + minDistance, currentBid + minDistance);
            Print("Adjusting sell SL from ", newSL, " to ", adjustedSL, " (min distance: ", minDistance, " pips)");
            Print("Current Bid: ", currentBid, ", Current Ask: ", currentAsk);
         }
      }
      
      // Additional validation: ensure adjusted SL is not too close to current price
      if(posType == POSITION_TYPE_BUY && adjustedSL >= currentBid)
      {
         Print("ERROR: Adjusted buy SL ", adjustedSL, " is above or equal to current bid ", currentBid);
         return;
      }
      else if(posType == POSITION_TYPE_SELL && adjustedSL <= currentAsk)
      {
         Print("ERROR: Adjusted sell SL ", adjustedSL, " is below or equal to current ask ", currentAsk);
         return;
      }
      
      // Only proceed if we have a valid stop loss
      if(isValidSL || adjustedSL != newSL)
      {
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         
         request.action = TRADE_ACTION_SLTP;
         request.position = positionTicket;  // Specify the position ticket
         request.symbol = _Symbol;
         request.sl = adjustedSL;
         request.tp = currentTP;     // Preserve existing take profit
         
         Print("=== Executing SL Modification ===");
         Print("Final SL: ", adjustedSL, " (original: ", newSL, ")");
         Print("Position: ", positionTicket, ", Type: ", (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"));
         
         if(OrderSend(request, result))
         {
            Print("SUCCESS: Stop loss modified to: ", adjustedSL, " for position ", positionTicket);
         }
         else
         {
            Print("FAILED: Stop loss modification. Error: ", result.retcode, " - ", result.comment);
            Print("Requested SL: ", newSL, ", Adjusted SL: ", adjustedSL, ", Current Price: ", currentPrice);
            Print("Stops Level: ", stopsLevel, ", Min Distance: ", minDistance);
         }
      }
      else
      {
         Print("ERROR: Stop loss ", newSL, " is too close to current price ", currentPrice, " for position ", positionTicket);
         Print("Minimum distance required: ", minDistance, " pips");
      }
   }
   else
   {
      Print("Failed to select position ", positionTicket, " for modification");
   }
}

//+------------------------------------------------------------------+
//| Count open trades for current symbol                          |
//+------------------------------------------------------------------+
int CountOpenTrades()
{
   int count = 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            count++;
         }
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| Check if current time is within trading hours                 |
//+------------------------------------------------------------------+
bool IsTradingTime()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   return (dt.hour >= StartHour && dt.hour < EndHour);
}

//+------------------------------------------------------------------+
//| Custom indicator function for Goldbach levels                 |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CHART_CHANGE)
   {
      // Redraw Goldbach levels on chart
      DrawGoldbachLevels();
   }
}

//+------------------------------------------------------------------+
//| Draw Goldbach levels on chart                                 |
//+------------------------------------------------------------------+
void DrawGoldbachLevels()
{
   // Remove existing objects
   ObjectsDeleteAll(0, "Goldbach_");
   
   // Draw Goldbach levels
   for(int i = 0; i < (int)Goldbach_Levels; i++)
   {
      string objName = "Goldbach_Level_" + IntegerToString(i);
      ObjectCreate(0, objName, OBJ_HLINE, 0, 0, g_goldbachLevels[i]);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrBlue);
      ObjectSetInteger(0, objName, OBJPROP_STYLE, STYLE_DOT);
      ObjectSetInteger(0, objName, OBJPROP_WIDTH, 1);
      ObjectSetString(0, objName, OBJPROP_TEXT, "Goldbach " + IntegerToString(i));
   }
   
   // Draw PO3 levels
   for(int i = 0; i < 3; i++)
   {
      string objName = "PO3_Level_" + IntegerToString(i);
      ObjectCreate(0, objName, OBJ_HLINE, 0, 0, g_po3Levels[i]);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrRed);
      ObjectSetInteger(0, objName, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, objName, OBJPROP_WIDTH, 2);
      ObjectSetString(0, objName, OBJPROP_TEXT, "PO3 " + IntegerToString(i));
   }
}
