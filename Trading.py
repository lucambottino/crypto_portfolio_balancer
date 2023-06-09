ó
Bòiac           @   s  d  d l  Z  d  d l Z d  d l Z d  d l Z d  d l Z d  d l Z d  d l Z d  d l Z d  d l m Z d  d l	 m	 Z	 d Z
 e j e
  Z d Z i d d 6d d	 6d
 d 6Z d Z d Z e j d e d d d e
 d e d e j  Z d Z d Z d d d     YZ d S(   iÿÿÿÿN(   t   Database(   t   Orderss9   %(asctime)s,%(msecs)d %(levelname)s %(name)s: %(message)ss   %Y-%b-%d %H:%M:%Ss	   debug.logt   debugs
   trades.logt   tradings   general.logt   errorss   binance-trader.logs,   %(asctime)-15s - %(levelname)s:  %(message)st   filenamet   filemodet   at   formatt   datefmtt   levelgü©ñÒMbP?gü©ñÒMb@?t   Tradingc           B   sò   e  Z d  Z d Z e Z e Z d  Z d  Z	 d  Z
 d  Z d  Z d  Z d Z d Z d Z d Z d Z e Z d   Z e d  Z d   Z d	   Z d
   Z d   Z d   Z d   Z d   Z d   Z d   Z d   Z  d   Z! d   Z" d   Z# RS(   i    i   gÉ?i   i   i   c         C   sÆ   d j  |  GH| |  _ |  j j |  _ |  j j |  _ |  j j |  _ |  j j |  _ |  j j |  _ |  j j |  _ |  j j	 |  _	 |  j j
 d k r t |  _
 n  |  j |  j j d |  j j |  _ d  S(   Ns   options: {0}t   TOKENR   (   R   t   optiont   orderidt   order_idt   quantityt	   wait_timet	   stop_losst
   increasingt
   decreasingt   amountt	   commisiont   TOKEN_COMMISIONt   setup_loggert   symbolR   t   logger(   t   selfR   (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   __init__J   s    	c         C   sh   t  j |  } t  j t j  } | rJ | j t  j  | j t  j  n  | j t  | j	 |  | S(   s*   Function setup as many loggers as you want(
   t   loggingt	   getLoggert   StreamHandlert   syst   stdoutt   setLevelt   DEBUGt   setFormattert	   formattert
   addHandler(   R   R   R   R   t   stout_handler(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyR   c   s    c         C   sÀ   |  j    yw t j | | |  } t j | | d | d | |  j j g  |  j j d | | | t	 |  | f  | |  _
 | SWn8 t k
 r» } |  j j d |  t j |  j  d  SXd  S(   Ni    t   BUYsE   %s : Buy order created id:%d, q:%.8f, p:%.8f, Take profit aprox :%.8fs   Buy error: %s(   t   check_orderR   t	   buy_limitR    t   writeR   t   profitR   t   infot   floatR   t	   ExceptionR   t   timet   sleept   WAIT_TIME_BUY_SELLt   None(   R   R   R   t   buyPricet   profitableSellingPricet   orderIdt   e(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   buyv   s    
()	c         C   s×  t  j | |  } | d d k rE | d d k rE |  j j d  n³ t j |  j  | d d k r | d d k r |  j j d  np | d d k rË | d d k rË |  j j d  |  j | |  n- |  j | |  |  j j d	  d
 |  _	 d St  j
 | | |  } | d } |  j j d |  t j |  j  | d d k r¶|  j j d |  |  j j d |  |  j j d |  j j t | d  | f  d
 |  _	 d |  _ d S|  j d
 k rÓt j |  j  |  j | | | |  rt  j | |  d d k rI|  j j d  qIn* |  j j d  |  j | |  t d  xr |	 d k r½t j |  j  t  j | |  d }	 t  j |  }
 |  j j d |	 |
 | f  |  j j d  qLWd
 |  _	 d |  _ n  d S(   s   
        The specified limit will try to sell until it reaches.
        If not successful, the order will be canceled.
        t   statust   FILLEDt   sideR(   s   Buy order filled... Try sell...s0   Buy order filled after 0.1 second... Try sell...t   PARTIALLY_FILLEDsA   Buy order partially filled... Try sell... Cancel remaining buy...s+   Buy order fail (Not filled) Cancel order...i    NR6   s   Sell order create id: %ds   Sell order (Filled) Id: %ds   LastPrice : %.8fs.   Profit: %%%s. Buy price: %.8f Sell price: %.8ft   prices   We apologize... Sold at loss...sU   We apologize... Cant sell even at loss... Please sell manually... Stopping program...i   s/   Status: %s Current price: %.8f Sell price: %.8fs   Sold! Continue trading...(   R   t	   get_orderR   R-   R0   R1   t   WAIT_TIME_CHECK_BUY_SELLt   cancelt   warningR   t
   sell_limitt   WAIT_TIME_CHECK_SELLR   R,   R.   R3   t
   order_dataR   t   stopt   exitt
   get_ticker(   R   R   R   R6   t
   sell_pricet
   last_pricet	   buy_ordert
   sell_ordert   sell_idt   sell_statust	   lastPrice(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   sell   sP       	
-		
	c         C   s¿  t  j | |  } |  j t | d   } | | |  j d } | d } | d k sb | d k r|  j | |  r| | k rt  j | |  }	 |  j j d |  |	 d }
 |	 t	 k rÀ t	 St
 j |  j  |	 d } | d k rÿ d GH|  j j d  t	 S|  j | |
  t Sqt  j | | |  }	 d	 | GHt
 j |  j  |	 d } | d k rcd GHt	 S|	 d }
 |  j | |
  t Sq»d
 GHt	 Sn+ | d k r·d |  _ d  |  _ d GHt	 St Sd  S(   NR=   id   R9   t   NEWR<   s   Stop-loss, sell market, %sR6   s   Stop-loss, solds   Stop-loss, sell limit, %ss?   Cancel did not work... Might have been sold before stop loss...R:   i    s   Order filled(   R   R>   t   calcR.   R   R@   t   sell_marketR   R-   t   TrueR0   R1   t   WAIT_TIME_STOP_LOSSt   FalseRB   R   R3   RD   (   R   R   R   R6   RI   t
   stop_ordert	   stoppricet	   losspriceR9   t   selloRL   t
   statusloss(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyRE   è   sL    


	

		c         C   s  |  j    d } t j |  j  xm| |  j k  rt j | |  } | d } t | d  } t | d  } t | d  |  _ | d }	 |  j	 j
 d | | d | | f  |	 d	 k r)|  j | |  r%t j | |  }
 |  j	 j
 d
  |
 d |  _ |
 |  _ |
 t k rPq&| d 7} q# qPq# |	 d k r_| d |  _ | |  _ |  j	 j
 d  Pq# |	 d k r|  j	 j
 d  Pq# | d 7} q# q# Wd  S(   Ni    R;   R=   t   origQtyt   executedQtyR9   s5   Wait buy order: %s id:%d, price: %.8f, orig_qty: %.8fR6   RP   s   Buy market orderi   R:   t   FilledR<   s   Partial filled(   R)   R0   R1   R2   t   MAX_TRADE_SIZER   R>   R.   t   buy_filled_qtyR   R-   R@   t
   buy_marketR   RD   RS   (   R   R   R6   R   t   trading_sizet   orderR;   R=   t   orig_qtyR9   t   buyo(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   check&  s@    


$	
	
c         C   sx   t  j | |  } | s. d |  _ d  |  _ t S| d d k sN | d d k rt t  j | |  d |  _ d  |  _ t Sd  S(   Ni    R9   RP   t	   CANCELLED(   R   R>   R   R3   RD   RS   t   cancel_order(   R   R   R6   R)   (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyR@   b  s    		 		c         C   sI   y% | | |  j  j d | |  j SWn t k
 rD } d | GHd  SXd  S(   Nid   s   Calc Error: %s(   R   R,   R   R/   (   R   t   lastBidR7   (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyRQ   q  s
    %	c         C   s    |  j  d k r t d  n  d  S(   Ni    i   (   R   RF   (   R   (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyR)   |  s    c         C   sI  |  j  } t j |  } t j |  \ } } | |  j } | |  j } |  j |  } |  j j d k r t	 |  j j
  } t	 |  j j  } | } n  |  j j rù |  j d k rù | | d d }	 |  j j d | | | | | |	 | | |  j f  n  |  j d k r­|  j d  k	 rQ|  j }
 |  j t	 |
 d   } | | k rQ| } qQn  |  j j d k rr|  j j } n  t j d |  j d | | |  j | | f  } | j   d  S| | k rË|  j j d	 k sõ| t	 |  j j
  k rE|  j j d k rE|  j j d
 j |  j j | |   |  j d k rE|  j | | | |  qEn  d  S(   Nt   rangei    i   g      Y@s]   price:%.8f buyprice:%.8f sellprice:%.8f bid:%.8f ask:%.8f spread:%.2f  Originalsellprice:%.8fR=   t   targett   argsR,   s/   MOde: {0}, Lastsk: {1}, Profit Sell Price {2}, (   R   R   RG   t   get_order_bookR   R   RQ   R   t   modeR.   t   buypricet	   sellpricet   printsR   R   R   R   RD   R3   t	   threadingt   ThreadRO   t   startR-   R   R8   (   R   R   R   RN   Rh   t   lastAskR4   t	   sellPriceR5   t
   spreadPercRb   t   newProfitableSellingPricet
   sellAction(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   action  s:    		7	-
*%c         C   s   d S(   Ni    (    (   R   (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   logicÓ  s    c         C   sZ   |  j  j } t j |  } | s> |  j j d  t d  n  d   | d D | d <| S(   Ns#   Invalid symbol, please try again...i   c         S   s   i  |  ] } | | d   q S(   t
   filterType(    (   t   .0t   item(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pys
   <dictcomp>â  s   	 t   filters(   R   R   R   t   get_infoR   t   errorRF   (   R   R   t   symbol_info(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyR~   Ö  s    c         C   s!   t  | t j t  |  |   S(   N(   R.   t   matht   floor(   R   R   t   stepSize(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   format_stepæ  s    c         C   sE  t  } |  j j } |  j   d } t j |  \ } } t j |  } t | d d  } t | d d  } t | d d  }	 t |  j j  }
 t | d d  } t | d d	  } t |  j j	  | k  rà | |  _	 n  t |  j j
  | k  r| |  _
 n  | |  j	 } |	 | }
 |
 |
 d
 d }
 |	 } |  j d k rR|  j | }
 n  |  j d k rm|  j }
 n  |  j |
 |  }
 | t |
  } |
 |  _ | |  _ |
 | k  rÐ|  j j d | |
 f  t } n  | | k  rÿ|  j j d | | f  t } n  | |	 k  r.|  j j d |	 | f  t } n  | sAt d  n  d  S(   NR~   t   LOT_SIZEt   minQtyt   PRICE_FILTERt   minPricet   MIN_NOTIONALt   minNotionalR   t   tickSizei
   id   i    s(   Invalid quantity, minQty: %.8f (u: %.8f)s'   Invalid price, minPrice: %.8f (u: %.8f)s-   Invalid notional, minNotional: %.8f (u: %.8f)i   (   RS   R   R   R~   R   Rl   RG   R.   R   R   R   R   R   t	   step_sizeR   R   RU   RF   (   R   t   validR   R~   Rh   Rt   RN   R   R   R   R   R   R   t   notional(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   validateé  sL    
					c         C   s¸  d } g  } |  j  j } d GHd GH|  j   d GHd | GHd |  j GHd |  j GH|  j  j d k rÄ |  j  j d k s |  j  j d k r d	 GHt d
  n  d GHd |  j  j f GHd |  j  j f GHn, d GHd |  j  j	 GHd |  j
 GHd |  j GHd GHt j   } x° | |  j  j k r³t j   } t j d |  j d | f  } | j |  | j   t j   } | | |  j k  rt j |  j | |  |  j  j d k r°| d
 } q°qqWd  S(   Ni    s'   Auto Trading for Binance.com @yasinkuyus   
s
   Started...s   Trading Symbol: %ss   Buy Quantity: %.8fs   Stop-Loss Amount: %sRi   s&   Please enter --buyprice / --sellprice
i   s   Range Mode Options:s   	Buy Price: %.8fs   	Sell Price: %.8fs   Profit Mode Options:s   	Preferred Profit: %0.2f%%s%   	Buy Price : (Bid+ --increasing %.8f)s%   	Sell Price: (Ask- --decreasing %.8f)Rj   Rk   (   R   R   R   R   R   Rm   Rn   Ro   RF   R,   R   R   R0   t   loopRq   Rr   Ry   t   appendRs   R   R1   (   R   t   cyclet   actionsR   t	   startTimet   actionTradert   endTime(    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   run3  sB    
	$
N($   t   __name__t
   __module__R   R3   RD   RS   t
   buy_filledt   sell_filledR_   t   sell_filled_qtyR   R   R   R   R2   R?   RC   RT   R^   t   BNB_COMMISIONR   R   R   R8   RO   RE   Re   R@   RQ   R)   Ry   Rz   R~   R   R   R   (    (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyR   '   s>   			X	>	<				R				J(    (   t   osR    R0   t   configRq   R   R   t   logging.handlersR    R   t   formater_strt	   FormatterR%   R	   t   LOGGER_ENUMt   LOGGER_FILEt   FORMATt   basicConfigt   INFOR   R   R   R   (    (    (    s6   /Users/lucabottino/Documents/binance-trader/Trading.pyt   <module>   s*   