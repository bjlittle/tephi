# Copyright Tephi contributors
#
# This file is part of Tephi and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `tephi._PlotCollection` class.

"""
# Import tephi test package first so that some things can be initialised
# before importing anything else.
import tephi.tests as tests

import unittest
from unittest import mock

from tephi import _PlotCollection


class Test__spec_singleton(tests.TephiTest):
    def setUp(self):
        self.axes = mock.sentinel.axes
        self.func = mock.sentinel.func
        self.kwargs = mock.sentinel.kwargs
        self.zoom = mock.sentinel.zoom
        self.step = 1
        self.spec = [(self.step, self.zoom)]
        self.stop = 10
        self.istop = range(10)
        self.expected = [
            self.axes,
            self.func,
            self.kwargs,
            self.step,
            self.zoom,
        ]

    @mock.patch("tephi._PlotGroup")
    def test(self, mocker):
        pc = _PlotCollection(
            self.axes, self.spec, self.stop, self.func, self.kwargs
        )
        self.assertEqual(len(pc.groups), 1)
        self.assertEqual(mocker.call_count, 1)
        args, kwargs = mocker.call_args
        item = set(range(self.step, self.stop + self.step, self.step))
        self.expected.append(item)
        self.assertEqual(args, tuple(self.expected))

    @mock.patch("tephi._PlotGroup")
    def test_minimum_clip(self, mocker):
        minimum = 5
        pc = _PlotCollection(
            self.axes,
            self.spec,
            self.stop,
            self.func,
            self.kwargs,
            minimum=minimum,
        )
        self.assertEqual(len(pc.groups), 1)
        self.assertEqual(mocker.call_count, 1)
        args, kwargs = mocker.call_args
        item = set(range(minimum, self.stop + self.step, self.step))
        self.expected.append(item)
        self.assertEqual(args, tuple(self.expected))

    @mock.patch("tephi._PlotGroup")
    def test_minimum(self, mocker):
        minimum = -5
        pc = _PlotCollection(
            self.axes,
            self.spec,
            self.stop,
            self.func,
            self.kwargs,
            minimum=minimum,
        )
        self.assertEqual(len(pc.groups), 1)
        self.assertEqual(mocker.call_count, 1)
        args, kwargs = mocker.call_args
        item = set(range(minimum, self.stop + self.step, self.step))
        self.expected.append(item)
        self.assertEqual(args, tuple(self.expected))

    @mock.patch("tephi._PlotGroup")
    def test_minimum_bad(self, mocker):
        minimum = self.stop + 1
        emsg = "Minimum value of {} exceeds maximum " "threshold {}".format(
            minimum, self.stop
        )
        with self.assertRaisesRegex(ValueError, emsg):
            _PlotCollection(
                self.axes,
                self.spec,
                self.stop,
                self.func,
                self.kwargs,
                minimum=minimum,
            )

    @mock.patch("tephi._PlotGroup")
    def test_iterable(self, mocker):
        pc = _PlotCollection(
            self.axes, self.spec, self.istop, self.func, self.kwargs
        )
        self.assertEqual(len(pc.groups), 1)
        self.assertEqual(mocker.call_count, 1)
        args, kwargs = mocker.call_args
        item = set(self.istop)
        self.expected.append(item)
        self.assertEqual(args, tuple(self.expected))

    @mock.patch("tephi._PlotGroup")
    def test_iterable_minimum_clip(self, mocker):
        minimum = 7
        pc = _PlotCollection(
            self.axes,
            self.spec,
            self.istop,
            self.func,
            self.kwargs,
            minimum=minimum,
        )
        self.assertEqual(len(pc.groups), 1)
        self.assertEqual(mocker.call_count, 1)
        args, kwargs = mocker.call_args
        item = set(self.istop[minimum:])
        self.expected.append(item)
        self.assertEqual(args, tuple(self.expected))

    @mock.patch("tephi._PlotGroup")
    def test_iterable_minimum_bad(self, mocker):
        minimum = self.istop[-1] + 1
        emsg = "Minimum value of {} exceeds all other values".format(minimum)
        with self.assertRaisesRegex(ValueError, emsg):
            _PlotCollection(
                self.axes,
                self.spec,
                self.istop,
                self.func,
                self.kwargs,
                minimum=minimum,
            )


if __name__ == "__main__":
    unittest.main()
